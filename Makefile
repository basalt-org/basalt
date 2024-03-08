.PHONY: test

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

housing:
	. .venv/bin/activate && mojo run -I . examples/housing.mojo

pyhousing:
	. .venv/bin/activate && python examples/housing.py

stat:
	. .venv/bin/activate && mojo run -I . examples/stat.mojo


test:
# 	mojo run -I . test/test_uuid.mojo
# 	mojo run -I . test/test_node.mojo
	mojo run -I . test/test_tensorutils.mojo
# 	mojo run -I . test/test_graph.mojo
	mojo run -I . test/test_ops.mojo
# 	mojo run -I . test/test_layers.mojo
	mojo run -I . test/test_backward.mojo
# 	mojo run -I . test/test_loss.mojo
# 	mojo run -I . test/test_regression.mojo
# 	mojo run -I . test/test_traits.mojo
# 	mojo run -I . test/test_activations.mojo
# 	mojo run -I . test/test_broadcasting.mojo
	mojo run -I . test/test_mlops.mojo
# 	. .venv/bin/activate && mojo run -I . test/test_conv.mojo
# 	. .venv/bin/activate && mojo run -I . test/test_pool.mojo
# 	. .venv/bin/activate && mojo run -I . test/test_models_torch.mojo