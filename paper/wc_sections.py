import re

txt = re.sub(r'(?m)^\s*%.*$', '', open('main.tex').read())
body = txt.split('\n' + r'\bibliography')[0]

floats = re.findall(r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', body, re.S)
prose = re.sub(r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', '', body, flags=re.S)
prose = prose.split(r'\maketitle')[-1]

flt = re.findall(r'(\\begin\{(?:table\*?|figure\*?)\}.*?\\end\{(?:table\*?|figure\*?)\})', body, re.S)
print('floats:', len(flt), 'words:', sum(len(f.split()) for f in flt))
for f in flt:
    lab = re.search(r'\\label\{([^}]*)\}', f)
    rows = f.count(r'\\')
    print(f'   {lab.group(1) if lab else "?":28s} lines~{rows:3d} words {len(f.split()):4d}')

print('prose words total:', len(prose.split()))
parts = re.split(r'(\\(?:sub)?section\*?\{[^}]*\})', prose)
for i in range(1, len(parts), 2):
    name = re.sub(r'\\(?:sub)?section\*?\{|\}', '', parts[i])
    print(f'   {name:38s} {len(parts[i+1].split()):5d}')
