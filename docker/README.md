# Running Docker Container for Jupyter Notebooks

### Build Docker Container

```bash
  sudo docker docker-compose -f docker-compose.yml build
```

### Run Docker Container

```bash
  sudo docker-compose -f docker-compose.yml up
```

### Open Jupyter Notebooks in Browser
Grab the URL from the output of the previous up command

Example:
```bash
  http://127.0.0.1:8888/?token=154a634851c1a1754e1e37a6d2466a289325af6986b1d868
```
