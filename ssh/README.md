# SSH Workspace on Modal

Launches an H100 container with SSH access and a persistent Modal Volume. Your local workspace is baked into the image, and changes made over SSH are synced to the volume every 30s. On the next run, files are restored from the volume automatically.

## Usage

Launch in detached mode:
```bash
modal run -d modal_train.py
```

Or with a custom SSH key:
```bash
modal run -d modal_train.py --key-path ~/.ssh/my_key.pub
```

Grab the SSH connection info from the output:
```
SSH available at: r433.modal.host:38057
Connect with: ssh root@r433.modal.host -p 38057
```

Files created or modified over SSH are synced to the volume in the background. On the next `modal run -d`, those files are synced back into the container automatically.

Then do all work over SSH:
```bash
ssh root@r433.modal.host -p 38057
```

Stream logs or stop the detached app using the app ID from `modal run -d` output:
```bash
modal app logs <app-id>
modal app stop <app-id>
```
