import random
import sqlite3

from PIL import Image

from examples.sqlite_search import create_index, fingerprint, main, search, store


def test_matches_exhaustive_hamming_search():
	rng = random.Random(42)
	query = rng.getrandbits(64)
	values = [rng.getrandbits(64) for _ in range(100)]
	# Include boundary cases with differences evenly spread over all four bands.
	for distance in range(65):
		bits = [16 * (i % 4) + i // 4 for i in range(distance)]
		values.append(query ^ sum(1 << bit for bit in bits))
	with sqlite3.connect(':memory:') as db:
		create_index(db)
		for i, value in enumerate(values):
			store(db, str(i), '%016x' % value)
		for radius in range(65):
			expected = sorted((bin(query ^ v).count('1'), str(i)) for i, v in enumerate(values)
			                  if bin(query ^ v).count('1') <= radius)
			assert search(db, '%016x' % query, radius) == expected


def test_image_search_and_reindex(tmp_path, capsys):
	db = tmp_path / 'images.sqlite'
	gallery = tmp_path / 'photos'
	gallery.mkdir()
	source = gallery / 'source.png'
	copy = tmp_path / 'resized.png'
	image = Image.new('L', (64, 64))
	image.putdata([(x * 3 + y * 5) % 256 for y in range(64) for x in range(64)])
	image.save(source)
	image.resize((128, 128)).save(copy)
	assert main(['index', str(db), str(gallery)]) == 0
	distance = bin(int(fingerprint(source), 16) ^ int(fingerprint(copy), 16)).count('1')
	assert distance < 8
	assert main(['search', str(db), str(copy), '--distance', str(distance)]) == 0
	assert capsys.readouterr().out.strip() == '{}\t{}'.format(distance, source)
	Image.new('L', (64, 64)).save(source)
	assert main(['index', str(db), str(source)]) == 0
	assert main(['search', str(db), str(copy), '--distance', str(distance)]) == 0
	assert capsys.readouterr().out == ''
