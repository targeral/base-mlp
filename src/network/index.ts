import * as tf from '@tensorflow/tfjs';

class Network {
  private sizes: number[];

  private num_layers: number;

  private biases: tf.Tensor<tf.Rank>[] = [];

  private weights: tf.Tensor<tf.Rank>[] = [];

  constructor(sizes: number[]) {
    this.sizes = sizes;
    this.num_layers = sizes.length;
    this.biases = sizes.slice(1).map(y => tf.randomNormal([y, 1]));
    // console.info('this.biases', this.biases);
    /**
     * [2 3 1]
     *
     * weights[0] = W = [
     *   [1 4]
     *   [2 5]
     *   [3 6]
     * ];
     *
     *
     * W[0][1] = 1 -> 表示第一层第二个节点到第二层的第一个节点的权重值
     * W（jk） 是链接第一层的k节点到第二层的j节点的权重值
     *
     * 第二层到第三层
     * weight[1] = [
     *  [7]
     *  [8]
     *  [9]
     * ]
     *
     * [2, 3]
     * [3, 1]
     *
     * (2, 3)
     * (3, 1)
     */
    this.weights = sizes
      .slice(0, -1)
      .map((x, i) => tf.randomNormal([sizes[i + 1], x]));
  }

  feedforward(a: tf.Tensor<tf.Rank>): tf.Tensor<tf.Rank> {
    // console.info('a');
    // a.print();
    for (let i = 0; i < this.num_layers - 1; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      // console.info('b');
      // b.print();
      // console.info('w');
      // w.print();

      a = sigmoid(tf.add(tf.matMul(w, a), b));
    }
    return a;
  }

  async SGD_v1(
    // [训练输入, 其对应的期望输出][]
    training_data: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
    epochs: number,
    eta: number,
  ) {
    // debugger;
    const n = training_data.length;
    // console.info('训练数据量为', n);
    for (let j = 0; j < epochs; j++) {
      console.info(`第${j + 1}训练开始:`);
      // training_data.forEach(td => {
      //   td[0].print();
      //   td[1].print();
      // });
      tf.util.shuffle(training_data);
      // training_data.forEach(td => {
      //   td[0].print();
      //   td[1].print();
      // });

      //损耗平均值
      const costAgv = await this.update_mini_batch(training_data, eta);
      console.info(`第${j + 1}训练完成,loss:`, costAgv);
      // console.info('weights')
      // this.weights.forEach(w => w.print());
    }
  }

  async update_mini_batch(
    mini_batch: [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][],
    eta: number,
  ) {
    let nabla_b = this.biases.map(b => tf.zerosLike(b));

    // TODO: remove: start
    // console.info('nabla_b', nabla_b);
    // nabla_b.forEach(b => {
    //   b.print();
    // });
    // TODO: remove: end

    let nabla_w = this.weights.map(w => tf.zerosLike(w));
    // TODO: remove: start
    // console.info('nabla_w', nabla_w);
    // nabla_w.forEach(w => {
    //   w.print();
    // });
    // TODO: remove: end
    let costValueSum = 0;
    for (const [input, expect] of mini_batch) {
      const [delta_nabla_b, delta_nabla_w, costValue] = await this.backprop(
        input,
        expect,
      );
      costValueSum += costValue;
      // TODO: remove start
      // console.info('delta_nabla_b', delta_nabla_b);
      // delta_nabla_b.forEach(b => b.print());
      // console.info('delta_nabla_w', delta_nabla_w);
      // delta_nabla_w.forEach(w => w.print());
      // TODO: remove end
      nabla_b = nabla_b.map((nb, i) => nb.add(delta_nabla_b[i]));
      nabla_w = nabla_w.map((nw, i) => {
        console.info(`====delta_nabla_w[${i}]====`, delta_nabla_w[i].arraySync())
        console.info(`====before add ${i} ====`);
        console.info(nw.arraySync());

        console.info(`====after add ${i} ====`);
        const ret = nw.add(delta_nabla_w[i]);
        console.info('ret', ret.arraySync());
        console.info(nw.arraySync());
        return ret;
      });
      console.info('after backprop');
      for (const [index, nw] of nabla_w.entries()) {
        console.info(`第${index+1}层到${index+2}层的权重`)
        console.info(await nw.array());
      }
    }

    // console.info('Cost Value:', costValueSum / mini_batch.length);

    // TODO: remove start
    // console.info('更新权重和编制的矩阵');
    // console.info('nabla_b', nabla_b);
    // nabla_b.forEach(b => b.print());
    // console.info('nabla_w', nabla_w);
    // nabla_w.forEach(w => w.print());
    // TODO: remove end

    this.weights = this.weights.map((w, i) => {
      const ret = tf.sub(w, tf.mul(eta / mini_batch.length, nabla_w[i]));
      console.info(i);
      console.info('nabla_w[i]');
      nabla_w[i].print();
      console.info('new weight');
      ret.print();
      return ret;
    });

    this.biases = this.biases.map((b, i) =>
      tf.sub(b, tf.mul(eta / mini_batch.length, nabla_b[i])),
    );

    return costValueSum / mini_batch.length;
  }

  // 后续研究
  async backprop(
    x: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
  ): Promise<[tf.Tensor<tf.Rank>[], tf.Tensor<tf.Rank>[], number]> {
    // console.info('backprop');
    // x.print(true);
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

    // 损失计算
    // 计算均方误差
    const mse = tf.losses.meanSquaredError(y, activation);
    // 获取损失值的张量并转换为 JavaScript 数值
    const mseValue = mse.dataSync()[0];
    // console.info('答案:');
    // y.print();
    // console.info('训练的结果:');
    // activation.print();
    // console.info('损失值:', mseValue);

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
    for (const [index, w] of nabla_w.entries()) {
      console.info(`第${index+1}层到${index+2}层的权重`)
        const ret = await w.array();
        console.info(ret)
    }
    return [nabla_b, nabla_w, mseValue];
  }

  async cost_derivative(
    output_activations: tf.Tensor<tf.Rank>,
    y: tf.Tensor<tf.Rank>,
  ): Promise<tf.Tensor<tf.Rank>> {
    return tf.sub(output_activations, y);
  }

  print() {
    for (let i = 0; i < this.num_layers - 1; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      console.info('b');
      b.print();
      console.info('w');
      w.print();
    }
  }

  store() {
    const weights = this.weights.map(w => w.arraySync());
    const biases = this.biases.map(b => b.arraySync());
    const jsonString = JSON.stringify({weights, biases});
    const blob = new Blob([jsonString], {type: 'application/json'});

    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    const filename = 'model.json';
    a.download = filename;

    a.click();

    URL.revokeObjectURL(url);
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

export { Network, sigmoid };
