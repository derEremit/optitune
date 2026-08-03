# Flatpak packaging (scaffold)

Files:

| File | Purpose |
|------|---------|
| `org.optitune.OptiTune.yml` | flatpak-builder manifest |
| `org.optitune.OptiTune.desktop` | Desktop entry |
| `org.optitune.OptiTune.metainfo.xml` | AppStream metadata |
| `../../assets/icon.svg` | App icon |

## Build (local)

Requires `flatpak`, `flatpak-builder`, and the Freedesktop 24.08 SDK/runtime:

```bash
# From repo root
flatpak install -y flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08
flatpak-builder --user --install --force-clean /tmp/optitune-flatpak-build \
  packaging/flatpak/org.optitune.OptiTune.yml
flatpak run org.optitune.OptiTune
```

## Notes

- PulseAudio / PipeWire access is enabled for mic input.
- First builds download SDKs; expect several minutes.
- CI may treat this job as non-blocking until the pip module set is hardened.
