import click
import cv2
import numpy as np

VERTEX_RADIUS = 10

class RoiSelector:
	def __init__(self, img, n_vertices):
		self.img = img
		self.working_img = img.copy()
		self.n_vertices = n_vertices
		self.vertices = []

	def close_to(self):
		pass

	def on_click(self, event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			for vertex in self.vertices:
				print(vertex)
				if np.linalg.norm([vertex[0] - x, vertex[1] - y]) < VERTEX_RADIUS:
					print('close')
					print(np.linalg.norm([vertex[0] - x, vertex[1] - y]))

			if len(self.vertices) < self.n_vertices:
				self.vertices += [[x, y]]
				cv2.circle(self.working_img, (x, y), VERTEX_RADIUS, (0,0,255), -1)
				#print('({}, {})'.format(x, y))

	def plot(self):
		cv2.namedWindow('display', cv2.WINDOW_NORMAL)
		cv2.setWindowProperty('display',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.setMouseCallback('display', self.on_click)	

		while True:
			plot_img = self.working_img.copy()
			cv2.putText(plot_img, '{}/{}'.format(len(self.vertices), self.n_vertices), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)

			cv2.imshow('display', plot_img)
			key = cv2.waitKey(1) & 0xFF

			if key == ord('r'):
				self.vertices = []
				self.working_img = self.img.copy()

			if key == ord('q'):
				cv2.destroyAllWindows()
				break

@click.command()
@click.option('-n', default=2)
def main(n):
	img = cv2.imread('0001.png', -1).astype(np.float32)
	img = img / img.max()
	
	RoiSelector(img, n).plot()

	
if __name__ == '__main__':
	main()