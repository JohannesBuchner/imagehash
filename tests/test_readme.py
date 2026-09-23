import os


def test_run():
	# test code in README.rst file
	# find any chunks after ::
	# which code lines, which start with <tab> >>>
	parent_dir = os.path.dirname(os.path.dirname(__file__))
	with open(os.path.join(parent_dir, 'README.rst')) as f:
		chunk = [line.replace('\t>>> ', '') for line in f if line.startswith('\t>>> ')]

	code = ''.join(chunk)
	print("running::\n" + code)
	print("result:", exec(code, {}, {}))
