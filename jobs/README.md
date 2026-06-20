# Job Launcher Layout

This repository used to keep most launch logic in dozens of top-level `*.sh`
and `*.pbs` files. The shared behavior now lives under `jobs/`:

- `jobs/lib/common.sh`: shared shell helpers for staging, env setup, logging,
  manifest generation, and LoRA alpha defaults.
- `jobs/launch/`: runtime entrypoints that execute one run, a manifest worker,
  or an arbitrary validation command.
- `jobs/pbs/`: reusable PBS bodies. Top-level `*.pbs` files remain as thin
  compatibility wrappers so existing submission commands still work.

The top-level scripts are intentionally preserved as stable entrypoints, but
the duplicated implementation now lives in one place.
