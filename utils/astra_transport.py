"""Same-origin DNS-address failover BEFORE any HTTP request bytes are sent.

This is network path recovery, not a model/provider fallback or POST retry.
TLS hostname/certificate checks stay enabled. Proxies retain stdlib behavior.
"""
from __future__ import annotations
import http.client
import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

POLICY = {'name':'same_origin_pre_http_tls_v1','per_address_timeout_s':15,
          'connect_budget_s':45,'retry_after_http_send':False,
          'certificate_verification':True,'redirects_allowed':False,
          'proxy_policy':'respect_environment_stdlib_no_direct_bypass'}
_PREFERRED = {}


class OriginHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self,*args,expected_origin,attempts,**kwargs):
        super().__init__(*args,**kwargs)
        self.expected_origin=expected_origin
        self.attempts=attempts

    def connect(self):
        # A configured proxy is not replaced by a direct route. CONNECT/SNI are
        # handled exactly by the standard library for that path.
        if self._tunnel_host or self.host!=self.expected_origin:
            self.attempts.append({'mode':'stdlib_proxy_connection','direct_address_failover':False})
            return super().connect()
        if self._context.verify_mode!=ssl.CERT_REQUIRED or not self._context.check_hostname:
            raise ValueError('Astra transport requires certificate and hostname verification')
        addresses=[];seen=set()
        for entry in socket.getaddrinfo(self.host,self.port,type=socket.SOCK_STREAM):
            family,kind,proto,_,address=entry
            key=(family,address)
            if key not in seen:
                addresses.append(entry);seen.add(key)
        preferred=_PREFERRED.get((self.host,self.port))
        addresses.sort(key=lambda e:(e[0],e[4])!=preferred)
        deadline=time.monotonic()+min(float(self.timeout),POLICY['connect_budget_s'])
        last_error=None
        for family,kind,proto,_,address in addresses:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            started=time.monotonic();raw=None
            attempt_deadline=min(deadline,started+POLICY['per_address_timeout_s'])
            record={'mode':'same_origin_direct','address':address[0],'family':family,
                    'sni':self.host,'attempt_phase':'connect_before_HTTP_send'}
            self.attempts.append(record)
            try:
                raw=socket.socket(family,kind,proto)
                raw.settimeout(max(.001,attempt_deadline-started))
                if self.source_address:raw.bind(self.source_address)
                raw.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                raw.connect(address)
                handshake_remaining=attempt_deadline-time.monotonic()
                if handshake_remaining<=0:raise TimeoutError('same-origin TCP/TLS address budget exhausted')
                raw.settimeout(handshake_remaining)
                tls=self._context.wrap_socket(raw,server_hostname=self.host)
                # Only now can HTTPConnection.send() send headers/body. It is
                # never retried by this method after successful connect().
                tls.settimeout(self.timeout)
                self.sock=tls
                record['tls_verified']=True
                record['elapsed_s']=round(time.monotonic()-started,6)
                _PREFERRED[self.host,self.port]=(family,address)
                return
            except ssl.SSLCertVerificationError:
                if raw is not None:raw.close()
                record.update(tls_verified=False,error_type='SSLCertVerificationError',
                              elapsed_s=round(time.monotonic()-started,6))
                raise
            except (OSError,ssl.SSLError) as exc:
                if raw is not None:raw.close()
                self.sock=None;last_error=exc
                record.update(tls_verified=False,error_type=type(exc).__name__,
                              elapsed_s=round(time.monotonic()-started,6))
        if last_error is not None:raise last_error
        raise TimeoutError('same-origin TLS connection budget exhausted')


class OriginHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self,origin,attempts):
        super().__init__(context=ssl.create_default_context())
        self.origin=origin;self.attempts=attempts

    def https_open(self,request):
        url=urllib.parse.urlsplit(request.full_url)
        if url.hostname!=self.origin or url.scheme!='https' or url.port not in (None,443):
            raise ValueError('Astra request changed its fixed origin')
        def factory(*args,**kwargs):
            return OriginHTTPSConnection(*args,expected_origin=self.origin,attempts=self.attempts,**kwargs)
        return self.do_open(factory,request,context=self._context,check_hostname=self._check_hostname)


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise urllib.error.HTTPError(req.full_url,code,'Astra endpoint redirect refused',headers,fp)


def astra_urlopen(request,timeout=120):
    origin=urllib.parse.urlsplit(request.full_url).hostname
    if origin!='litellm.dex-gem.ai':raise ValueError('unexpected Astra origin')
    attempts=[]
    opener=urllib.request.build_opener(OriginHTTPSHandler(origin,attempts),RejectRedirects())
    try:
        response=opener.open(request,timeout=timeout)
        response.astra_connection_attempts=attempts
        return response
    finally:
        # No token, request payload or credentials are logged.
        print('[astra-transport] '+json.dumps({'policy':POLICY['name'],'connection_attempts':attempts}),flush=True)
