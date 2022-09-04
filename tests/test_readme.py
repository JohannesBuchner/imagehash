import os
import six


def test_run():
	# test code in README.rst file
	# find any chunks after ::
	# which code lines, which start with <tab> >>>
	chunk = None
	for line in open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.rst')):
		if not line.startswith('\t') and chunk is not None:
			# print("end of code segment. Have:", chunk)
			if len(chunk) > 0:
				code = ''.join(chunk)
				print("running::\n" + code)
				print("result:", six.exec_(code, {}, {}))
				chunk = None
		elif line.endswith('::\n'):
			# print("start of code segment:", line)
			chunk = []
		elif chunk is not None and line.startswith('\t>>> '):
			# print("appending:", line.replace('\t>>> ', ''))
			chunk.append(line.replace('\t>>> ', ''))