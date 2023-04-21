package djlsam.translators;

import ai.djl.ndarray.NDArray;

public record SamRawOutput(NDArray iouPred, NDArray lowResLogits, NDArray mask) {
	public void close() {
		iouPred.close();
		lowResLogits.close();
		mask.close();
	}
}
