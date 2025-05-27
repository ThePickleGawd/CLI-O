import docker
import time

start = time.time()

client = docker.from_env()

# Run container
container = client.containers.run(
    image="python:3.12",
    command='python -c "print(\'Hello, world!\')"',
    remove=True,       # Auto-remove after execution
    stdout=True,
    stderr=True
)

print(container.decode())
print(f"Total Time: {time.time() - start:.2f}")