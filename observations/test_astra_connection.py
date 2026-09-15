"""Credential-free CPU tests of the actual Astra connection/send boundary."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils import runtime


HOST = 'litellm.dex-gem.ai'
SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}


def addresses(count=2):
    # Documentation-only addresses; every socket and DNS call is mocked.
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
             (f'192.0.2.{i+1}', 443)) for i in range(count)]


def response_bytes(*, status=200, model=runtime.ASTRA_MODEL, group=runtime.ASTRA_MODEL,
                   completion='completed', result=None, error=None, invalid_json=False):
    raw = {'model': model, 'status': completion, 'error': error,
           'output': [{'type': 'message', 'role': 'assistant', 'content': [
               {'type': 'output_text', 'text': json.dumps({'ok': True} if result is None else result)}]}]}
    body = b'invalid-json' if invalid_json else json.dumps(raw).encode()
    return (f'HTTP/1.1 {status} Fixture\r\nContent-Length: {len(body)}\r\n'
            f'x-litellm-model-group: {group}\r\nLocation: https://other.invalid/\r\n\r\n').encode() + body


class FakeSocket:
    def __init__(self, response=None, connect_error=None, handshake_error=None, send_error_at=None,
                 read_error=None):
        self.response = response if response is not None else response_bytes()
        self.connect_error = connect_error
        self.handshake_error = handshake_error
        self.send_error_at = send_error_at
        self.read_error = read_error
        self.sent = []
        self.timeouts = []
        self.closed = False
        self.stream = None

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, address):
        self.address = address
        if self.connect_error:
            raise self.connect_error

    def do_handshake(self):
        if self.handshake_error:
            raise self.handshake_error

    def sendall(self, data):
        self.sent.append(bytes(data))
        if len(self.sent) == self.send_error_at:
            raise OSError('synthetic secret must not appear in audit')

    def makefile(self, mode):
        if self.read_error:
            raise self.read_error
        self.stream = io.BytesIO(self.response)
        return self.stream

    def close(self):
        self.closed = True


class AstraConnectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        credential = self.root/'fixture-key'
        credential.write_text('synthetic-secret-only')
        credential.chmod(0o600)
        self.stack.enter_context(patch.dict(os.environ, {'EGO_ASTRA_CREDENTIAL_FILE': str(credential)}))
        self.image = self.root/'evidence.png'
        self.image.write_bytes(b'original-image-fixture')
        self.dns = self.stack.enter_context(patch('utils.runtime.socket.getaddrinfo', return_value=addresses()))
        self.factory = self.stack.enter_context(patch('utils.runtime.socket.socket'))
        self.context = Mock()
        self.stack.enter_context(patch('utils.runtime.ssl.create_default_context', return_value=self.context))
        self.context.wrap_socket.side_effect = lambda sock, **kw: sock

    def request(self, audit='audit'):
        return runtime.astra_request('fixture prompt', SCHEMA, {'frame_ids': [0]}, [self.image],
                                     audit_dir=self.root/audit)

    def diagnostics(self, audit='audit'):
        return json.loads((self.root/audit/'connection_attempts.json').read_text())

    def test_bad_address_then_good_before_exact_single_post(self):
        bad = FakeSocket(connect_error=TimeoutError('synthetic-secret-only'))
        good = FakeSocket()
        self.factory.side_effect = [bad, good]
        result, provenance = self.request()
        self.assertEqual(result, {'ok': True})
        self.assertTrue(bad.closed and good.closed)
        self.assertTrue(good.stream.closed)
        self.assertEqual(bad.sent, [])
        wire = b''.join(good.sent)
        headers, body = wire.split(b'\r\n\r\n', 1)
        self.assertEqual(wire.count(b'POST /v1/responses HTTP/1.1'), 1)
        self.assertIn(b'Host: ' + HOST.encode(), headers)
        self.assertEqual(body, (self.root/'audit/effective_payload.json').read_bytes())
        image_url = json.loads(body)['input'][0]['content'][-1]['image_url']
        self.assertEqual(base64.b64decode(image_url.split(',', 1)[1]), self.image.read_bytes())
        self.assertEqual(good.timeouts[-1], 120)
        self.context.wrap_socket.assert_called_once_with(good, server_hostname=HOST,
                                                        do_handshake_on_connect=False)
        self.assertTrue(self.context.check_hostname)
        self.assertEqual(self.context.verify_mode, ssl.CERT_REQUIRED)
        self.context.set_alpn_protocols.assert_called_once_with(['http/1.1'])
        self.dns.assert_called_once_with(HOST, 443, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        diagnostic = self.diagnostics()
        self.assertEqual([a['outcome'] for a in diagnostic['attempts']], ['failed', 'connected'])
        self.assertEqual(diagnostic['request_sha256'], provenance['effective_payload_sha256'])
        plan = runtime.astra_stage_inputs('fixture prompt', SCHEMA, {'frame_ids': [0]}, [self.image])
        self.assertEqual(plan['connection_policy'], provenance['connection_policy'])
        self.assertEqual(plan['connection_policy'], diagnostic['policy'])
        for path in (self.root/'audit').iterdir():
            self.assertNotIn('synthetic-secret', path.read_text())
            self.assertNotIn('Authorization', path.read_text())

    def test_tls_hostname_mismatch_closes_without_http_then_tries_next(self):
        bad = FakeSocket(handshake_error=ssl.SSLCertVerificationError('hostname mismatch'))
        good = FakeSocket()
        self.factory.side_effect = [bad, good]
        self.request()
        self.assertTrue(bad.closed)
        self.assertEqual(bad.sent, [])
        self.assertEqual(self.diagnostics()['attempts'][0]['error_type'], 'SSLCertVerificationError')
        self.assertEqual(self.context.wrap_socket.call_count, 2)

    def test_real_tls_rejects_wrong_hostname_and_untrusted_certificate(self):
        # In-memory TLS only. Ephemeral self-signed test key, never a credential.
        cert, key = self.root/'test-cert.pem', self.root/'test-key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt',
                        'ec_paramgen_curve:prime256v1', '-nodes', '-days', '1',
                        '-subj', f'/CN={HOST}', '-addext', f'subjectAltName=DNS:{HOST}',
                        '-keyout', str(key), '-out', str(cert)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cert, key)
        for host, trusted in ((HOST, True), ('wrong.invalid', True), (HOST, False)):
            with self.subTest(host=host, trusted=trusted):
                client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                if trusted:
                    client_context.load_verify_locations(cafile=str(cert))
                self.dns.return_value = addresses(1)
                raw_socket = FakeSocket()
                self.factory.return_value = raw_socket
                def wrap(raw, *, server_hostname, do_handshake_on_connect):
                    self.assertFalse(do_handshake_on_connect)
                    client_in, client_out = ssl.MemoryBIO(), ssl.MemoryBIO()
                    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
                    client = client_context.wrap_bio(client_in, client_out,
                                                     server_hostname=server_hostname)
                    server = server_context.wrap_bio(server_in, server_out, server_side=True)
                    def handshake():
                        # Each iteration exchanges actual TLS records, never waits/polls.
                        for _ in range(10):
                            try:
                                client.do_handshake()
                                return
                            except ssl.SSLWantReadError:
                                pass
                            server_in.write(client_out.read())
                            try:
                                server.do_handshake()
                            except ssl.SSLWantReadError:
                                pass
                            client_in.write(server_out.read())
                        raise AssertionError('in-memory TLS failed to exchange handshake records')
                    raw.do_handshake = handshake
                    return raw
                diagnostics = {'attempts': []}
                with patch('utils.runtime.ssl.create_default_context', return_value=client_context), \
                     patch.object(client_context, 'wrap_socket', side_effect=wrap):
                    if host == HOST and trusted:
                        connected = runtime._connect_astra(host, 443, diagnostics)
                        connected.close()
                        self.assertEqual(diagnostics['attempts'][0]['outcome'], 'connected')
                    else:
                        with self.assertRaises(ConnectionError):
                            runtime._connect_astra(host, 443, diagnostics)
                        self.assertEqual(diagnostics['attempts'][0]['error_type'], 'SSLCertVerificationError')
                self.assertTrue(raw_socket.closed)
                self.assertEqual(raw_socket.sent, [])

    def test_distinct_wrapped_socket_cleanup_on_tls_failure(self):
        raw = FakeSocket()
        tls = FakeSocket(handshake_error=ssl.SSLError('fixture'))
        self.dns.return_value = addresses(1)
        self.factory.return_value = raw
        self.context.wrap_socket.side_effect = None
        self.context.wrap_socket.return_value = tls
        with self.assertRaises(ConnectionError):
            self.request()
        self.assertTrue(raw.closed and tls.closed)
        self.assertEqual(tls.sent, [])

    def test_all_addresses_exhausted_deduplicated_and_audited(self):
        self.dns.return_value = addresses() + addresses()
        sockets = [FakeSocket(connect_error=OSError('synthetic-secret-only')) for _ in range(2)]
        self.factory.side_effect = sockets
        with self.assertRaisesRegex(ConnectionError, 'before POST'):
            self.request()
        self.assertEqual(self.factory.call_count, 2)
        self.assertTrue(all(s.closed and not s.sent for s in sockets))
        self.assertEqual(self.diagnostics()['phase'], 'connect_exhausted')
        self.assertFalse(json.loads((self.root/'audit/transport_failure.json').read_text())['retry'])

    def test_total_budget_limits_attempts_and_tcp_tls_share_address_budget(self):
        now = [0.]
        self.stack.enter_context(patch('utils.runtime.time.monotonic', side_effect=lambda: now[0]))
        self.dns.return_value = addresses(10)
        made = []
        def make(*args):
            sock = FakeSocket()
            def connect(address):
                now[0] += 2
            def handshake():
                now[0] += 3
                raise TimeoutError('fixture')
            sock.connect = connect
            sock.do_handshake = handshake
            made.append(sock)
            return sock
        self.factory.side_effect = make
        with self.assertRaises(ConnectionError):
            self.request()
        self.assertEqual(len(made), 4)
        self.assertEqual(now[0], 20)
        self.assertTrue(all(s.timeouts == [5, 3, 3] and s.closed and not s.sent for s in made))

    def test_late_handshake_success_is_closed_without_post(self):
        now = [0.]
        self.stack.enter_context(patch('utils.runtime.time.monotonic', side_effect=lambda: now[0]))
        self.dns.return_value = addresses(1)
        sock = FakeSocket()
        def handshake():
            now[0] = 5
        sock.do_handshake = handshake
        self.factory.return_value = sock
        with self.assertRaises(ConnectionError):
            self.request()
        self.assertTrue(sock.closed)
        self.assertEqual(sock.sent, [])

    def test_remaining_total_budget_caps_last_address(self):
        now = [0.]
        self.stack.enter_context(patch('utils.runtime.time.monotonic', side_effect=lambda: now[0]))
        def resolve(*args):
            now[0] = 18
            return addresses()
        self.dns.side_effect = resolve
        sock = FakeSocket()
        def connect(address):
            now[0] += 1
        def handshake():
            now[0] += 1
            raise TimeoutError('fixture')
        sock.connect = connect
        sock.do_handshake = handshake
        self.factory.return_value = sock
        with self.assertRaises(ConnectionError):
            self.request()
        self.factory.assert_called_once()
        self.assertEqual(sock.timeouts, [2, 1, 1])
        self.assertTrue(sock.closed)
        self.assertEqual(sock.sent, [])

    def test_blocking_dns_overrun_prevents_any_socket(self):
        now = [0.]
        self.stack.enter_context(patch('utils.runtime.time.monotonic', side_effect=lambda: now[0]))
        def resolve(*args):
            now[0] = 21
            return addresses()
        self.dns.side_effect = resolve
        with self.assertRaises(ConnectionError):
            self.request()
        self.factory.assert_not_called()
        self.assertEqual(self.diagnostics()['dns_elapsed_seconds'], 21)

    def test_dns_failure_and_empty_answers_never_send(self):
        for index, answers in enumerate((socket.gaierror('fixture'), [])):
            with self.subTest(answers=answers):
                self.dns.side_effect = answers if isinstance(answers, Exception) else None
                self.dns.return_value = answers
                with self.assertRaises(OSError):
                    self.request(str(index))
                self.factory.assert_not_called()

    def test_ipv6_resolver_address_preserved(self):
        address = ('2001:db8::1', 443, 0, 0)
        self.dns.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', address)]
        sock = FakeSocket()
        self.factory.return_value = sock
        self.request()
        self.assertEqual(sock.address, address)
        self.factory.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP)

    def test_no_replay_after_header_send_body_send_or_read_failure(self):
        for index, kwargs in enumerate(({'send_error_at': 1}, {'send_error_at': 2},
                                       {'read_error': TimeoutError('synthetic-secret-only')})):
            with self.subTest(kwargs=kwargs):
                self.factory.reset_mock()
                sock = FakeSocket(**kwargs)
                self.factory.return_value = sock
                with self.assertRaises(OSError):
                    self.request(str(index))
                self.factory.assert_called_once()
                self.assertTrue(sock.closed)
                self.assertNotIn('synthetic-secret-only', json.dumps(self.diagnostics(str(index))))

    def test_redirect_http_error_json_model_group_completion_schema_never_retry(self):
        cases = ({'status': 302}, {'status': 307}, {'status': 429}, {'status': 503},
                 {'invalid_json': True}, {'model': 'other'}, {'group': 'other'},
                 {'completion': 'incomplete'}, {'error': {'message': 'failed'}}, {'result': {'ok': 1}})
        for index, kwargs in enumerate(cases):
            with self.subTest(kwargs=kwargs):
                self.factory.reset_mock()
                sock = FakeSocket(response=response_bytes(**kwargs))
                self.factory.return_value = sock
                with self.assertRaises(Exception):
                    self.request(str(index))
                self.factory.assert_called_once()
                self.assertEqual(b''.join(sock.sent).count(b'POST /v1/responses HTTP/1.1'), 1)
                self.assertTrue(sock.closed and sock.stream.closed)

    def test_wrap_failure_and_cancellation_close_raw_socket(self):
        for index, error in enumerate((ssl.SSLError('fixture'), KeyboardInterrupt())):
            sock = FakeSocket()
            self.factory.return_value = sock
            self.dns.return_value = addresses(1)
            self.context.wrap_socket.side_effect = error
            with self.assertRaises(ConnectionError if isinstance(error, OSError) else KeyboardInterrupt):
                self.request(str(index))
            self.assertTrue(sock.closed)
            self.assertEqual(sock.sent, [])

    def test_http_connection_cannot_implicitly_reconnect(self):
        connection = Mock()
        connection.request.side_effect = OSError('send failure')
        sock = FakeSocket()
        self.factory.return_value = sock
        with patch('utils.runtime.http.client.HTTPSConnection', return_value=connection):
            with self.assertRaises(OSError):
                self.request()
        self.assertEqual(connection.auto_open, 0)
        connection.request.assert_called_once()
        connection.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
