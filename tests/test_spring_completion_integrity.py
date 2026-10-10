"""Reject damaged/truncated downloads before decoder access."""
import base64,importlib.util
from pathlib import Path
import google_crc32c
import pytest
spec=importlib.util.spec_from_file_location('completion',Path(__file__).parents[1]/'scripts/complete_spring_supplement.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def fixture():
    payload=b'\0'*12+(32).to_bytes(4,'little')+b'abcdefghijklmnop'
    crc=base64.b64encode(google_crc32c.Checksum(payload).digest()).decode()
    return payload,{'bytes':32,'hash':'md5=unused, crc32c='+crc}

def test_valid_header_and_crc():
    data,info=fixture(); assert m.checked_payload(data,info)

def test_corruption_rejected():
    data,info=fixture()
    for broken in [data[:-1],data[:-1]+b'X',data[:12]+(30).to_bytes(4,'little')+data[16:]]:
        with pytest.raises(ValueError): m.checked_payload(broken,info)
