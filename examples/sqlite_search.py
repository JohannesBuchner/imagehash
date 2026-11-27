"""Local reverse image search using 64-bit pHash and SQLite. See README.rst."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

from PIL import Image
import imagehash


def fingerprint(path):
	with Image.open(path) as image:
		return str(imagehash.phash(image, hash_size=8))


def bands(digest):
	return [int(digest[i:i + 4], 16) for i in range(0, 16, 4)]


def create_index(db):
	db.execute("""CREATE TABLE IF NOT EXISTS phash64 (
		path TEXT PRIMARY KEY, digest TEXT NOT NULL,
		b0 INTEGER NOT NULL, b1 INTEGER NOT NULL,
		b2 INTEGER NOT NULL, b3 INTEGER NOT NULL)""")
	for i in range(4):
		db.execute('CREATE INDEX IF NOT EXISTS phash64_b{0} ON phash64(b{0})'.format(i))


def store(db, path, digest):
	db.execute('INSERT OR REPLACE INTO phash64 VALUES (?, ?, ?, ?, ?, ?)', (path, digest, *bands(digest)))


def search(db, digest, radius):
	if not 0 <= radius <= 64:
		raise ValueError('distance must be between 0 and 64')
	sql = 'SELECT path, digest FROM phash64'
	params = []
	if radius < 8:
		# At least one of four bands differs in at most floor(radius / 4) bits.
		conditions = []
		for i, band in enumerate(bands(digest)):
			neighbors = [band]
			if radius >= 4:
				neighbors.extend(band ^ (1 << bit) for bit in range(16))
			conditions.append('b{} IN ({})'.format(i, ','.join('?' for _ in neighbors)))
			params.extend(neighbors)
		sql += ' WHERE ' + ' OR '.join(conditions)
	query = int(digest, 16)
	matches = []
	for path, candidate in db.execute(sql, params):
		distance = bin(query ^ int(candidate, 16)).count('1')
		if distance <= radius:
			matches.append((distance, path))
	return sorted(matches)


def image_paths(inputs):
	extensions = Image.registered_extensions()
	for path in inputs:
		if path.is_dir():
			for child in path.rglob('*'):
				if child.suffix.lower() in extensions and child.is_file():
					yield child
		else:
			yield path


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest='command', required=True)
	index = commands.add_parser('index', help='add or refresh files or directories recursively')
	index.add_argument('database', type=Path)
	index.add_argument('images', nargs='+', type=Path)
	query = commands.add_parser('search', help='find images within a Hamming distance')
	query.add_argument('database', type=Path)
	query.add_argument('image', type=Path)
	query.add_argument('--distance', type=int, default=4)
	args = parser.parse_args(argv)
	try:
		if args.command == 'index':
			with closing(sqlite3.connect(str(args.database))) as db, db:
				create_index(db)
				for path in image_paths(args.images):
					store(db, str(path.resolve()), fingerprint(path))
		else:
			if not 0 <= args.distance <= 64:
				parser.error('--distance must be between 0 and 64')
			uri = args.database.resolve().as_uri() + '?mode=ro'
			with closing(sqlite3.connect(uri, uri=True)) as db:
				for distance, path in search(db, fingerprint(args.image), args.distance):
					print('{}\t{}'.format(distance, path))
	except (OSError, sqlite3.Error) as error:
		print(error, file=sys.stderr)
		return 1
	return 0


if __name__ == '__main__':
	sys.exit(main())
