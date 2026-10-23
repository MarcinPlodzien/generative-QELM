"""Repository paths and atomic JSON/provenance utilities."""
from pathlib import Path
import hashlib,json
CODE=Path(__file__).resolve().parents[1]
ROOT=CODE.parent
CACHE_DIR=CODE/'cache'
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as file:
  for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
