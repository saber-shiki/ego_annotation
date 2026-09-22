"""No-network tests for address failover, origin/TLS locking and no POST retry."""
import io
import socket
import ssl
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch
from utils.astra_transport import OriginHTTPSConnection, OriginHTTPSHandler, RejectRedirects, astra_urlopen, _PREFERRED

HOST='litellm.dex-gem.ai'
ADDRESSES=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('192.0.2.1',443)),
           (socket.AF_INET,socket.SOCK_STREAM,6,'',('192.0.2.2',443))]


class AddressTransportTests(unittest.TestCase):
    def setUp(self):_PREFERRED.clear()

    def connection(self,attempts,context=None):
        return OriginHTTPSConnection(HOST,443,timeout=120,context=context or ssl.create_default_context(),
            expected_origin=HOST,attempts=attempts)

    def test_failed_tls_tries_other_same_host_before_sending(self):
        raw1=Mock();raw2=Mock();tls=Mock();attempts=[]
        with patch('socket.getaddrinfo',return_value=ADDRESSES),patch('socket.socket',side_effect=[raw1,raw2]),\
             patch.object(ssl.SSLContext,'wrap_socket',side_effect=[TimeoutError('fixture'),tls]) as wrap:
            conn=self.connection(attempts);conn.connect()
        self.assertIs(conn.sock,tls);self.assertEqual(len(attempts),2)
        self.assertEqual([c.kwargs['server_hostname'] for c in wrap.call_args_list],[HOST,HOST])
        raw1.close.assert_called_once();raw1.sendall.assert_not_called();raw2.sendall.assert_not_called()
        self.assertTrue(attempts[1]['tls_verified']);tls.settimeout.assert_called_once_with(120)
        self.assertEqual(_PREFERRED[HOST,443],(socket.AF_INET,('192.0.2.2',443)))

    def test_recent_verified_address_preferred_only_while_in_dns(self):
        _PREFERRED[HOST,443]=(socket.AF_INET,('192.0.2.2',443))
        with patch('socket.getaddrinfo',return_value=ADDRESSES),patch('socket.socket') as factory,\
             patch.object(ssl.SSLContext,'wrap_socket',return_value=Mock()):
            self.connection([]).connect()
            factory.return_value.connect.assert_called_once_with(('192.0.2.2',443))
        with patch('socket.getaddrinfo',return_value=ADDRESSES[:1]),patch('socket.socket') as factory,\
             patch.object(ssl.SSLContext,'wrap_socket',return_value=Mock()):
            self.connection([]).connect()
            factory.return_value.connect.assert_called_once_with(('192.0.2.1',443))

    def test_certificate_failure_is_not_retried(self):
        with patch('socket.getaddrinfo',return_value=ADDRESSES),patch('socket.socket') as factory,\
             patch.object(ssl.SSLContext,'wrap_socket',side_effect=ssl.SSLCertVerificationError('fixture')):
            with self.assertRaises(ssl.SSLCertVerificationError):self.connection([]).connect()
            self.assertEqual(factory.call_count,1)

    def test_insecure_context_rejected_before_network(self):
        context=ssl._create_unverified_context()
        with patch('socket.getaddrinfo') as dns:
            with self.assertRaisesRegex(ValueError,'verification'):self.connection([],context).connect()
            dns.assert_not_called()

    def test_proxy_is_not_bypassed(self):
        conn=self.connection([]);conn.set_tunnel(HOST,443)
        with patch('http.client.HTTPSConnection.connect') as standard,patch('socket.getaddrinfo') as dns:
            conn.connect();standard.assert_called_once();dns.assert_not_called()

    def test_budget_bounds_connection_attempts(self):
        clock=[0.]
        def fail(*args,**kwargs):
            clock[0]=46.
            raise TimeoutError('fixture')
        with patch('socket.getaddrinfo',return_value=ADDRESSES),patch('socket.socket') as factory, \
             patch('utils.astra_transport.time.monotonic',side_effect=lambda:clock[0]), \
             patch.object(ssl.SSLContext,'wrap_socket',side_effect=fail):
            with self.assertRaises(TimeoutError):self.connection([]).connect()
            self.assertEqual(factory.call_count,1)

    def test_http_send_or_response_error_never_reopens_post(self):
        opener=Mock();opener.open.side_effect=ConnectionResetError('after send fixture')
        request=urllib.request.Request('https://'+HOST+'/v1/responses',data=b'body',method='POST')
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaises(ConnectionResetError):astra_urlopen(request,timeout=120)
        opener.open.assert_called_once_with(request,timeout=120)

    def test_origin_redirect_and_port_cannot_change(self):
        with self.assertRaisesRegex(ValueError,'origin'):
            astra_urlopen(urllib.request.Request('https://other.invalid/v1/responses'))
        handler=OriginHTTPSHandler(HOST,[])
        with self.assertRaisesRegex(ValueError,'origin'):
            handler.https_open(urllib.request.Request('https://'+HOST+':444/v1/responses'))
        with self.assertRaises(urllib.error.HTTPError):
            RejectRedirects().redirect_request(urllib.request.Request('https://'+HOST+'/v1/responses'),
                io.BytesIO(),307,'redirect',{},'https://other.invalid/')


if __name__=='__main__':unittest.main()
