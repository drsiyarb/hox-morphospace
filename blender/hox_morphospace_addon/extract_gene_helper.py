"""
Helper script for Hox Gene Expression Painter add-on.
Extracts a single gene's expression from the h5ad file and appends it to the CSV.

Usage:  python extract_gene_helper.py <gene_name>
        python extract_gene_helper.py <gene1> <gene2> ...

Called automatically by the Blender add-on when user types a custom gene name.
Can also be run standalone to batch-add genes.
"""
import sys
import os
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(SCRIPT_DIR, "s_fca_biohub_body_10x.h5ad")
CSV_PATH   = os.path.join(SCRIPT_DIR, "blender_gene_expression.csv")
HOX_GENES  = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']


def extract_gene(gene_name):
    """Extract normalized expression for a gene and append to CSV."""
    import scanpy as sc
    from sklearn.preprocessing import MinMaxScaler

    print(f"Loading h5ad ...", file=sys.stderr)
    adata = sc.read_h5ad(H5AD_PATH)
    gn = list(adata.var_names)

    # Check gene exists
    if gene_name not in gn:
        # Try case-insensitive search
        matches = [g for g in gn if g.lower() == gene_name.lower()]
        if matches:
            gene_name = matches[0]
            print(f"  Matched as: {gene_name}", file=sys.stderr)
        else:
            # Partial match
            partials = [g for g in gn if gene_name.lower() in g.lower()]
            if partials:
                print(f"ERROR: '{gene_name}' not found. Did you mean: {partials[:10]}",
                      file=sys.stderr)
            else:
                print(f"ERROR: '{gene_name}' not found in dataset ({len(gn)} genes)",
                      file=sys.stderr)
            return False

    # Check if already in CSV
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        existing_cols = reader.fieldnames
        if gene_name in existing_cols:
            print(f"'{gene_name}' already in CSV, skipping", file=sys.stderr)
            return True

    print(f"Extracting {gene_name} ...", file=sys.stderr)

    # Get Hox mask (same as original export)
    hox_idx = [gn.index(g) for g in HOX_GENES]
    X_raw = adata.X
    X_raw = X_raw.toarray() if hasattr(X_raw, 'toarray') else X_raw
    hox_mask = X_raw[:, hox_idx].sum(1) > 0

    # Extract gene expression for Hox+ cells
    gi = gn.index(gene_name)
    raw_vals = X_raw[hox_mask, gi].astype(float)
    if hasattr(raw_vals, 'toarray'):
        raw_vals = raw_vals.toarray().ravel()

    # Normalize to [0, 1]
    vmin, vmax = raw_vals.min(), raw_vals.max()
    if vmax > vmin:
        norm_vals = (raw_vals - vmin) / (vmax - vmin)
    else:
        norm_vals = np.zeros_like(raw_vals)

    n_pos = int((raw_vals > 0).sum())
    n_total = len(norm_vals)
    print(f"  {gene_name}: {n_pos}/{n_total} expressing", file=sys.stderr)

    # Read existing CSV
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    if len(rows) != n_total:
        print(f"ERROR: CSV has {len(rows)} rows but expected {n_total}", file=sys.stderr)
        return False

    # Append new column
    headers.append(gene_name)
    for i, row in enumerate(rows):
        row[gene_name] = f"{norm_vals[i]:.6f}"

    # Write back
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Appended '{gene_name}' to {CSV_PATH}", file=sys.stderr)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_gene_helper.py <gene_name> [gene2 ...]")
        print()
        print("Extracts gene expression from h5ad and appends to CSV.")
        print("The Blender add-on calls this automatically.")
        print()
        print("Examples:")
        print("  python extract_gene_helper.py en")
        print("  python extract_gene_helper.py nub dac eya")
        sys.exit(1)

    genes = sys.argv[1:]
    success = 0
    for gene in genes:
        if extract_gene(gene):
            success += 1

    print(f"\nDone: {success}/{len(genes)} genes extracted", file=sys.stderr)
    sys.exit(0 if success == len(genes) else 1)
