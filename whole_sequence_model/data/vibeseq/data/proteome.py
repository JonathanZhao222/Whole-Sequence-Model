import gzip
import urllib.request
from pathlib import Path

HUMAN_PROTEOME_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/Eukaryota/UP000005640/"
    "UP000005640_9606.fasta.gz"
)

STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def download_proteome(url: str, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def parse_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    proteins: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    proteins[current_id] = "".join(current_seq)
                current_id = line.split("|")[1] if "|" in line else line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        proteins[current_id] = "".join(current_seq)

    return proteins


def filter_standard(proteins: dict[str, str]) -> dict[str, str]:
    return {pid: seq for pid, seq in proteins.items() if set(seq) <= STANDARD_AAS}
