// import clsx from 'clsx';
import * as tf from '@tensorflow/tfjs';
import Generator from '@/components/Generator';
import { Network } from '@/network';
import { trainData } from './trainData';
import { transformJsonData2TrainData } from '@/utils/data';

import './index.css';
import { useEffect } from 'react';
// import style from './style.module.css';

const net = new Network([100, 16, 16, 10]);

const Index = (): JSX.Element => {
  useEffect(() => {
    console.info('a', tf.tensor([1]).equal(tf.tensor([1])));
    const main = async () => {
      const training_data = transformJsonData2TrainData(trainData);
      await net.SGD_v1(training_data, 3);
      console.info('complete');
    };
    main();
  }, []);

  const handleTestData = async (
    testData: { data: number[][]; target: number[]; content: string }[],
  ) => {
    const testTensorData = transformJsonData2TrainData(testData);
    console.info('==============');
    console.info(await testTensorData[0][1].data());
    console.info('==============');
    const testResults = await Promise.all(
      testTensorData.map(async ([x, y]) => {
        const t = await net.feedforward(x);
        console.info('输出 t');
        t.print();
        console.info(y);
        const a = tf.argMax(t);
        a.print();
        const aData = await a.data();
        console.info('aData', aData, aData[0]);
        const b = await tf.argMax(y);
        b.print();
        const bData = await b.data();
        console.info('bData', bData, bData[0]);
        return [aData[0], bData[0]];
      }),
    );
    console.info('testResults', testResults);
    const accuracy = testResults.reduce(
      (sum, [x, y]) => sum + (x === y ? 1 : 0),
      0,
    );
    console.info('accuracy', accuracy);
    // const output = net.feedforward(test);
    // output.print();
    // const outputData = await output.data();
    // console.info('outputData', outputData);
  };

  return (
    <div className="container-box">
      <Generator onTestData={handleTestData}></Generator>
      <br />
    </div>
  );
};

export default Index;
