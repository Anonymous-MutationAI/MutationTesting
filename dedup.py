import json, glob, shutil
for p in sorted(glob.glob('results/v2/*.jsonl')):
    seen, out, dup = set(), [], 0
    for line in open(p):
        r = json.loads(line)
        key = (r['type'], r.get('idx'), tuple(r['obl']) if 'obl' in r else None)
        if r['type'] != 'meta' and key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(line)
    if dup:
        shutil.copy(p, p + '.bak')
        open(p, 'w').writelines(out)
    print(f"{p}: {dup} duplicate records removed")
