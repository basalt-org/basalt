setup: venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r python-requirements.txt

venv:
	test -d .venv || python3 -m venv .venv

clean:
	rm -rf .venv

package:
	mojo package dainemo -o dainemo.📦

mnist:
	. .venv/bin/activate && mojo run -I . examples/mnist.mojo

pymnist:
	. .venv/bin/activate && python examples/mnist.py