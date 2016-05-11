# Create train samples
x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)

# Create network with 2 layers and random initialized
net = nl.net.newff([[-7, 7]],[5, 1])

