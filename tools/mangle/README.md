### Mangle (mg) tooling

- example.mg: simple facts and a sibling rule, prints all siblings.
- run_mangle.sh: wrapper to execute a .mg file via mg.

Usage:
```bash
./tools/mangle/run_mangle.sh ./tools/mangle/example.mg
```
Pass mg flags after `--`:
```bash
./tools/mangle/run_mangle.sh ./tools/mangle/example.mg -- -root .
```
