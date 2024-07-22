#BOXPLOT
import numpy as np
import matplotlib.pyplot as plt
data=np.random.rand(20,3)
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()

#HEATMAP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data=np.random.rand(8,6)
sns.heatmap(data,annot=True,fmt='.2f',cmap='coolwarm')
plt.title("Heatmap")
plt.show()

#CONTOURPLOT
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-5,5,100)
y=np.linspace(-5,5,100)
X,Y=np.meshgrid(x,y)
Z=np.sin(np.sqrt(X**2+Y**2))
plt.contour(X,Y,Z,cmap='viridis',levels=20)
plt.title("Contourplot")
plt.show()

#3D SURFACEPLOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
x=np.linspace(-5,5,100)
y=np.linspace(-5,5,100)
X,Y=np.meshgrid(x,y)
Z=np.sin(np.sqrt(X**2+Y**2))
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,cmap='viridis')
plt.title('3D Surfaceplot')
plt.show()
