#!/usr/bin/env python3
"""Download 100 molecules from ChEMBL and save as SDF."""

import urllib.request
import sys

url = "https://www.ebi.ac.uk/chembl/api/data/molecule.sdf?molecule_properties__mw_freebase__lte=500&molecule_properties__mw_freebase__gte=200&limit=100"

print(f"Downloading from: {url}")
try:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
        print(f"Downloaded {len(data)} bytes")

        with open("chembl_100.sdf", "wb") as f:
            f.write(data)
        print("Saved to chembl_100.sdf")

        # Count molecules (each ends with $$$$)
        content = data.decode('utf-8', errors='ignore')
        n_mols = content.count('$$$$')
        print(f"Contains {n_mols} molecules")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
