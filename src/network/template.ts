import * as tf from '@tensorflow/tfjs';

class Network {
  private num_layers: number;

  private sizes: number[];

  private biases: tf.Tensor<tf.Rank>[] = [];

  private weights: tf.Tensor<tf.Rank>[] = [];

  constructor(sizes: number[]) {
    this.num_layers = sizes.length;
    this.sizes = sizes;
    this.biases = sizes.slice(1).map(y => tf.randomNormal([y, 1]));
    this.weights = sizes
      .slice(0, -1)
      .map((x, i) => tf.randomNormal([sizes[i + 1], x]));
  }

  feedforward(a: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    for (let i = 0; i < this.num_layers - 1; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      a = sigmoid(tf.add(tf.matMul(w, a), b));
    }
    return a;
  }

  async SGD(
    training_data: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
    epochs: number,
    mini_batch_size: number,
    eta: number,
    test_data?: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
  ) {
    const n = training_data.length;
    const n_test = test_data ? test_data.length : 0;

    for (let j = 0; j < epochs; j++) {
      tf.util.shuffle(training_data);
      const mini_batches = Array.from(
        { length: Math.ceil(n / mini_batch_size) },
        (_, k) =>
          training_data.slice(k * mini_batch_size, (k + 1) * mini_batch_size),
      );

      for (const mini_batch of mini_batches) {
        await this.update_mini_batch(mini_batch, eta);
      }

      if (test_data) {
        const accuracy = await this.evaluate(test_data);
        console.log(`Epoch ${j}: ${accuracy} / ${n_test}`);
      } else {
        console.log(`Epoch ${j} complete`);
      }
    }
  }

  async update_mini_batch(
    mini_batch: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
    eta: number,
  ) {
    const nabla_b = this.biases.map(b => tf.zerosLike(b));
    const nabla_w = this.weights.map(w => tf.zerosLike(w));

    for (const [x, y] of mini_batch) {
      const [delta_nabla_b, delta_nabla_w] = await this.backprop(x, y);
      nabla_b.forEach((nb, i) => nb.add(delta_nabla_b[i]));
      nabla_w.forEach((nw, i) => nw.add(delta_nabla_w[i]));
    }

    this.weights = this.weights.map((w, i) =>
      tf.sub(w, tf.mul(eta / mini_batch.length, nabla_w[i])),
    );

    this.biases = this.biases.map((b, i) =>
      tf.sub(b, tf.mul(eta / mini_batch.length, nabla_b[i])),
    );
  }

  async backprop(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
  ): Promise<[tf.Tensor<tf.Rank>[], tf.Tensor<tf.Rank>[]]> {
    const nabla_b = this.biases.map(b => tf.zerosLike(b));
    const nabla_w = this.weights.map(w => tf.zerosLike(w));

    // Feedforward
    let activation = x;
    const activations = [activation];
    const zs: tf.Tensor<tf.Rank>[] = [];

    for (let i = 0; i < this.num_layers - 1; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      const z = tf.add(tf.matMul(w, activation), b);
      zs.push(z);
      activation = sigmoid(z);
      activations.push(activation);
    }

    // Backward pass
    let delta = tf.mul(
      await this.cost_derivative(activations[activations.length - 1], y),
      sigmoid_prime(zs[zs.length - 1]),
    );

    nabla_b[nabla_b.length - 1] = delta;
    nabla_w[nabla_w.length - 1] = tf.matMul(
      delta,
      tf.transpose(activations[activations.length - 2]),
    );

    for (let l = 2; l < this.num_layers; l++) {
      const z = zs[zs.length - l];
      const sp = sigmoid_prime(z);
      delta = tf.matMul(
        tf.transpose(this.weights[this.weights.length - l + 1]),
        delta,
      );
      delta = tf.mul(delta, sp);
      nabla_b[nabla_b.length - l] = delta;
      nabla_w[nabla_w.length - l] = tf.matMul(
        delta,
        tf.transpose(activations[activations.length - l - 1]),
      );
    }

    return [nabla_b, nabla_w];
  }

  async evaluate(
    test_data: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
  ): Promise<number> {
    const testResults = await Promise.all(
      test_data.map(async ([x, y]) => [
        tf.argMax(await this.feedforward(x).data()),
        y,
      ]),
    );
    const accuracy = testResults.reduce(
      (sum, [x, y]) => sum + (x === y ? 1 : 0),
      0,
    );
    return accuracy;
  }

  async cost_derivative(
    output_activations: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
  ): Promise<tf.Tensor<tf.Rank>> {
    return tf.sub(output_activations, y);
  }
}

// Miscellaneous functions
function sigmoid(z: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
  return tf.div(1.0, tf.add(1.0, tf.exp(tf.neg(z))));
}

function sigmoid_prime(z: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
  const sigmoid_z = sigmoid(z);
  return tf.mul(sigmoid_z, tf.sub(1.0, sigmoid_z));
}

export { Network, sigmoid, sigmoid_prime };
