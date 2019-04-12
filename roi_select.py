import click
import cv2
import numpy as np

VERTEX_RADIUS = 5

class RoiSelector:
	def __init__(self, img, max_vertices):
		self.main_window = 'Selector'
		self.crop_window = 'Cropped Region'
		self.img = img.copy()
		self.max_vertices = max_vertices
		self.vertices = []
		self.n_vertices = 0
		self.move_vertex = None
		self.additional_vertices = 0

	@staticmethod
	def close_to(v1, v2):
		return np.linalg.norm([v1[0] - v2[0], v1[1] - v2[1]]) < VERTEX_RADIUS

	def on_click(self, event, x, y, flags, params):
		if self.move_vertex is not None:
			self.vertices[self.move_vertex] = [x, y]
		if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
			if self.move_vertex is None:
				for idx, vertex in enumerate(self.vertices):
					if self.close_to([x, y], vertex):
						self.move_vertex = idx
			else:
				self.vertices[self.move_vertex] = [x, y]
				self.move_vertex = None
		elif event == cv2.EVENT_LBUTTONDOWN and self.move_vertex is None:
			if self.n_vertices < self.max_vertices + self.additional_vertices:
				self.vertices += [[x, y]]
				self.n_vertices += 1


	def plot(self):
		working_img = self.img.copy()

		for idx, vertex in enumerate(self.vertices):
			if idx != self.move_vertex:
				cv2.circle(working_img, (vertex[0], vertex[1]), VERTEX_RADIUS, (0,0,255), -1)
			else:
				cv2.circle(working_img, (vertex[0], vertex[1]), VERTEX_RADIUS, (200,0,200), -1)

		if self.n_vertices > 1:
			vertices = self.vertices.copy()
			vertices += [self.vertices[0]]
			for v1, v2 in zip(vertices[:-2], vertices[1:-1]):
				cv2.line(working_img, (v1[0], v1[1]), (v2[0], v2[1]),(255,0,0),2)
			if self.n_vertices == self.max_vertices + self.additional_vertices:
				v1, v2 = self.vertices[-1], self.vertices[0]
				cv2.line(working_img, (v1[0], v1[1]), (v2[0], v2[1]),(255,0,0),2)

		cv2.putText(working_img, '{}/{}'.format(len(self.vertices), self.max_vertices + self.additional_vertices), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 3)
		cv2.imshow(self.main_window, working_img)

	def get_vertices(self):
		return self.vertices

	def get_mask(self):
		mask = np.zeros_like(self.img)
		ignore_mask_color = (255,) * self.img.shape[2]
		cv2.fillPoly(mask, np.array([self.vertices], dtype=np.int32), ignore_mask_color)

		return mask

	def run(self):
		cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
		cv2.setWindowProperty(self.main_window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.setMouseCallback(self.main_window, self.on_click)	

		while True:
			self.plot()
			key = cv2.waitKey(1) & 0xFF

			if key == ord('r'):
				self.vertices = []
				self.n_vertices = 0
				self.move_vertex = None
				self.additional_vertices = 0
				cv2.destroyWindow(self.crop_window)

			if key == ord('a'):
				self.additional_vertices += 1
				cv2.destroyWindow(self.crop_window)

			if key == ord('s'):
				if self.additional_vertices > 0:
					if self.n_vertices == self.max_vertices + self.additional_vertices:
						self.vertices = self.vertices[:-1]
						self.n_vertices -= 1

					self.additional_vertices -= 1
					cv2.destroyWindow(self.crop_window)

			if key == ord('m'):
				if self.n_vertices == self.max_vertices + self.additional_vertices:
					cv2.destroyAllWindows()
					return self.get_mask()

			if key == ord('d'):
				cv2.destroyWindow(self.crop_window)				

			if key == ord('c'):
				if self.n_vertices == self.max_vertices + self.additional_vertices:
					mask = self.get_mask()
					masked_image = cv2.bitwise_and(self.img, mask)
					cv2.namedWindow(self.crop_window, cv2.WINDOW_NORMAL)
					cv2.setWindowProperty(self.crop_window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
					cv2.imshow(self.crop_window, masked_image)

			if key == ord('q'):
				cv2.destroyAllWindows()
				break

@click.command()
@click.option('-n', default=3, type=click.IntRange(3))
def main(n):
	img = cv2.imread('0001.png', -1).astype(np.float32)
	img = (img / img.max() * 255).astype(np.uint8)
	
	RoiSelector(img, n).run()

if __name__ == '__main__':
	main()