# Systemd user units

Copy to `~/.config/systemd/user/` on the VPS and enable:

```bash
scp deploy/systemd/eidolon-housing-*.{service,timer} hostinger:.config/systemd/user/
ssh hostinger 'systemctl --user daemon-reload &&
  systemctl --user enable --now eidolon-housing-extract.timer eidolon-housing-trend.timer'
```

`eidolon-index.timer` (the 10-minute `build`) predates this directory and
already lives on the host; these two share its lock and skip politely.
