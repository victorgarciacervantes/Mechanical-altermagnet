import re
from pathlib import Path

p = Path('master_thesis_slides.tex')
lines = p.read_text(encoding='utf-8').splitlines()
stack = []
errors = []
for i, line in enumerate(lines, 1):
    line2 = line.split('%', 1)[0]
    for m in re.finditer(r'\\begin\{([^}]+)\}|\\end\{([^}]+)\}', line2):
        env = m.group(1) or m.group(2)
        if m.group(1):
            stack.append((env, i))
        else:
            if not stack:
                errors.append((i, env, 'unmatched end'))
            else:
                top, top_line = stack[-1]
                if top != env:
                    errors.append((i, env, f'expected {top} from line {top_line}'))
                    break
                stack.pop()
print('ERRORS')
for item in errors:
    print(item)
print('OPEN')
for env, line in stack:
    print(line, env)
