package djlsam.translators;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Transform;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.List;

/**
 * Translator for the SAM model.
 */
public class SamTranslator implements Translator<Image, SamRawOutput> {

	private final Builder builder;

	public SamTranslator(Builder builder) {
		this.builder = builder;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * @param ctx   Context for the translation
	 * @param input Input to the model (image)
	 * @return NDList of the input
	 */
	@Override
	public NDList processInput(TranslatorContext ctx, Image input) {
		// Convert image to NDArray
		NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
		// Apply build transforms
		for (Transform transform : builder.transforms) {
			array = transform.transform(array);
		}
		return new NDList(array);
	}

	/**
	 * @param ctx  Context for the translation
	 * @param list NDList of the output from the model
	 * @return SamRawOutput of the output from the model
	 */
	@Override
	public SamRawOutput processOutput(TranslatorContext ctx, NDList list) {
		// Note: this causes a memory leak if the NDArrays are not closed
		list.detach();
		NDArray iouPred = list.get(0);
		NDArray lowResLogits = list.get(1);
		NDArray mask = list.get(2);
		return new SamRawOutput(iouPred, lowResLogits, mask);
	}

	public static class Builder {
		public List<Transform> transforms;

		public Builder() {
			this.transforms = new java.util.ArrayList<Transform>();
		}

		public Builder addTransform(Transform transform) {
			this.transforms.add(transform);
			return this;
		}

		public SamTranslator build() {
			return new SamTranslator(this);
		}
	}
}
