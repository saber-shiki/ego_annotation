import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None
from observations.s3_cache import encoded,immutable_put,publish,read_manifest,restore,sha_file


class FakeS3:
    def __init__(self): self.objects={};self.puts=0
    def head_object(self, Bucket, Key):
        if Key not in self.objects: raise ClientError({'Error':{'Code':'404'}},'HeadObject')
        return {'ContentLength':len(self.objects[Key])}
    def put_object(self, Bucket, Key, Body, ContentLength, Metadata, IfNoneMatch):
        assert IfNoneMatch=='*'
        if Key in self.objects: raise ClientError({'Error':{'Code':'PreconditionFailed'}},'PutObject')
        data=Body.read() if hasattr(Body,'read') else Body
        assert len(data)==ContentLength
        self.objects[Key]=data;self.puts+=1
    def get_object(self, Bucket, Key):return {'Body':io.BytesIO(self.objects[Key])}


@unittest.skipIf(ClientError is None, 'install optional storage dependencies')
class S3CacheTests(unittest.TestCase):
    def test_dedup_and_remote_tamper(self):
        c=FakeS3();data=b'bytes';h=hashlib.sha256(data).hexdigest()
        immutable_put(c,'b','key',data,h,len(data))
        immutable_put(c,'b','key',data,h,len(data))
        self.assertEqual(c.puts,1)
        c.objects['key']=b'other'
        with self.assertRaisesRegex(ValueError,'remote content'):immutable_put(c,'b','key',data,h,len(data))
        self.assertEqual(c.puts,1)

    def test_publish_then_restore_preprocessing(self):
        with TemporaryDirectory() as d:
            root=Path(d)/'prep';root.mkdir()
            (root/'inspection.json').write_text(json.dumps({'source_unchanged':True,'missing':[],
                'status':'complete_preprocessing_readback','source_manifest_sha256':'x','source_manifest':'/original/bundle.json'}))
            (root/'array').write_bytes(b'observed values')
            c=FakeS3();receipt=publish(c,'bucket',root,'sample','preprocessing','cache',2)
            self.assertTrue(receipt['all_objects_readback_verified'])
            target=Path(d)/'restored';restore(c,'bucket',receipt['manifest_key'],target)
            for p in root.iterdir():self.assertEqual(p.read_bytes(),(target/p.name).read_bytes())
            with self.assertRaises(FileExistsError):restore(c,'bucket',receipt['manifest_key'],target)
            c.objects[receipt['manifest_key']]+=b' '
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):read_manifest(c,'bucket',receipt['manifest_key'])

    def test_no_manifest_for_failure_and_no_symlink(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            (root/'inspection.json').write_text(json.dumps({'status':'failed'}))
            c=FakeS3()
            with self.assertRaisesRegex(ValueError,'successful'):publish(c,'b',root,'case','preprocessing','cache',1)
            self.assertFalse(c.objects)

    def test_no_relocation_or_path_traversal(self):
        c=FakeS3();m={'schema':'ego.annotation.s3_artifact_package.v1','kind':'observations','original_root':'/original','files':[]}
        body=encoded(m);key=hashlib.sha256(body).hexdigest()+'.json';c.objects[key]=body
        with self.assertRaisesRegex(ValueError,'absolute paths'):restore(c,'b',key,'/different')
        m['files']=[{'path':'../escape','object_key':'a','sha256':'a'}]
        body=encoded(m);key=hashlib.sha256(body).hexdigest()+'.json';c.objects[key]=body
        with self.assertRaisesRegex(ValueError,'unsafe path'):read_manifest(c,'b',key)


if __name__=='__main__':unittest.main()
