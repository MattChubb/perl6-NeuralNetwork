from sklearn.datasets import make_moons

coords, values = make_moons(200, noise=0.20)

outfile = open('example_data', 'w')

for i in range(200):
  x, y = coords[i]
  v = values[i]
  csv = ','.join((
    str(x), str(y), str(v)
  ))
  outfile.writelines([csv, "\n"])

outfile.close()