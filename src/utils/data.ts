import * as tf from '@tensorflow/tfjs';

export const transformData2Tensor = (input: number[][]) => {
  return tf.reshape(input, [input.length * input[0].length, 1]);
};

export const transformJsonData2TrainData = (
  jsonData: {
    data: number[][];
    target: number[];
    content: string;
  }[],
): [tf.Tensor<tf.Rank>, tf.Tensor<tf.Rank>][] => {
  return jsonData.map(item => {
    const trainData = transformData2Tensor(item.data);
    const expectData = tf.reshape(item.target, [item.target.length, 1]);
    return [trainData, expectData];
  });
};

export const getExpectData = (expectNum: string | number) => {
  switch (Number(expectNum)) {
    case 0:
      return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    case 1:
      return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    case 2:
      return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    case 3:
      return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    case 4:
      return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
    case 5:
      return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    case 6:
      return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    case 7:
      return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0];
    case 8:
      return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0];
    case 9:
      return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    default:
      throw new Error('Not found expect data')
  }
};
