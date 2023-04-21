package djlsam;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import djlsam.translators.SamRawOutput;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class SamTest {

	/**
	 * Test the SAM model by loading an image and predicting the mask.
	 */
	@Test
	void predict() {
		Path imageFile = Paths.get("src/resources/images/test.jpg");
		ImageFactory factory = ImageFactory.getInstance();

		Sam sam;
		sam = new Sam();
		assertNotNull(sam);

		try {
			Image img = factory.fromFile(imageFile);
			SamRawOutput output = sam.predict(img);
			NDArray mask = output.mask();
			assertNotNull(mask);

			Image imgOut = img.duplicate().resize(1024, 1024, true);
			int height = (int) mask.getShape().get(0);
			int width = (int) mask.getShape().get(1);
			int[] pixels = new int[width * height];

			// We convert the mask to an image to visualize it
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int red;
					int green;
					int blue;
					int opacity;
					if (mask.getBoolean(h, w)) {
						red = 0;
						green = 0;
						blue = 255;
						opacity = 120;
					} else {
						red = 0;
						green = 0;
						blue = 0;
						opacity = 0;
					}
					int color = opacity << 24 | red << 16 | green << 8 | blue;
					pixels[h * width + w] = color; // black
				}
			}

			Image maskImage = factory.fromPixels(pixels, width, height);

			imgOut.drawImage(maskImage, true);

			Path outputPath = Paths.get("src/resources/images/test_out.png");
			imgOut.save(Files.newOutputStream(outputPath), "png");

			output.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}