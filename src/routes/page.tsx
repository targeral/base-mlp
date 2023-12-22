// import clsx from 'clsx';
import * as tf from '@tensorflow/tfjs';
import Generator from '@/components/Generator';
import { Network } from '@/network';
import { trainData } from './trainData';
import { transformJsonData2TrainData } from '@/utils/data';
import { a, b } from './temp-data';

import './index.css';
import { useEffect, useState } from 'react';
// import style from './style.module.css';

const net = new Network([100, 16, 16, 10]);

const Index = (): JSX.Element => {
  const [finish, setFinish] = useState(false);
  useEffect(() => {
    // console.info('a', tf.tensor([1]).equal(tf.tensor([1])));
    const main = async () => {
      const training_data = transformJsonData2TrainData(trainData);
      await net.SGD_v1(training_data, 78, 10);
      console.info('complete');
      setFinish(true);
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
        console.info('输出 t', t.arraySync());
        console.info('期望', y.arraySync());
        const a = tf.argMax(t);
        console.info('a', a.arraySync)
        const aData = await a.data();
        console.info('aData', aData, aData[0]);
        const b = await tf.argMax(y);
        console.info(b.arraySync())
        const bData = await b.data();
        console.info('bData', bData, bData[0]);
        return [aData[0], bData[0]];
      }),
    );
    console.info('testResults', testResults);
    console.info(`猜测是 ${testResults[0][0]}`, `实际是${testResults[0][1]}`)
    // const accuracy = testResults.reduce(
    //   (sum, [x, y]) => sum + (x === y ? 1 : 0),
    //   0,
    // );
    // console.info('accuracy', accuracy);
    // const output = net.feedforward(test);
    // output.print();
    // const outputData = await output.data();
    // console.info('outputData', outputData);
  };

  const storeModel = () => {
    net.store();
  }

  return (
    <div className="container-box">
      <Generator onTestData={handleTestData}></Generator>
      {finish ? 'hello network' : 'traning'}
      <br />
      <button onClick={storeModel}>存储模型</button>
    </div>
  );
};

export default Index;
