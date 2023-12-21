import React, { useState, useRef, useEffect } from 'react';
import { getExpectData } from '@/utils/data';

const DrawingBoard = (props: {onTestData?: (testData: { data: number[][]; content: string; target: number[] }[]) => void}) => {
  const [drawing, setDrawing] = useState(false);
  const [drawingText, setDrawingText] = useState('');
  const [dataRecords, setDataRecords] = useState<
    { data: number[][]; content: string }[]
  >([]);
  const [canvasSize, setCanvasSize] = useState({ width: 10, height: 10 });
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const pixelSize = 20; // Size of each "pixel"

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    contextRef.current = context;
  }, []);

  useEffect(() => {
    // Update canvas size when canvasSize changes
    const canvas = canvasRef.current;
    canvas.width = canvasSize.width * pixelSize;
    canvas.height = canvasSize.height * pixelSize;
    clearCanvas();
  }, [canvasSize]);

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    const x = Math.floor(offsetX / pixelSize) * pixelSize;
    const y = Math.floor(offsetY / pixelSize) * pixelSize;
    contextRef.current.beginPath();
    contextRef.current.rect(x, y, pixelSize, pixelSize);
    contextRef.current.fillStyle = 'black';
    contextRef.current.fill();
    setDrawing(true);
  };

  const draw = ({ nativeEvent }) => {
    if (!drawing) {
      return;
    }
    const { offsetX, offsetY } = nativeEvent;
    const x = Math.floor(offsetX / pixelSize) * pixelSize;
    const y = Math.floor(offsetY / pixelSize) * pixelSize;
    contextRef.current.beginPath();
    contextRef.current.rect(x, y, pixelSize, pixelSize);
    contextRef.current.fillStyle = 'black';
    contextRef.current.fill();
  };

  const finishDrawing = () => {
    setDrawing(false);
    recordData();
  };

  const handleDrawingTextInput = event => {
    console.info(event.target.value);
    setDrawingText(event.target.value);
  };

  const recordData = () => {
    const image = contextRef.current.getImageData(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height,
    );
    console.info(image);
    const imageData = image.data;

    const pixelData = [];
    for (let y = 0; y < canvasRef.current.height; y += pixelSize) {
      for (let x = 0; x < canvasRef.current.width; x += pixelSize) {
        const pixelIndex = (y * canvasRef.current.width + x) * 4;
        const grayValue = imageData[pixelIndex + 3] / 255; // Use red channel as grayscale
        // console.info('grayValue', grayValue);
        pixelData.push(grayValue); // Invert grayscale to match the requested format
      }
    }
    const newDataRecord = {
      data: reshapeArray(pixelData, canvasRef.current.width / pixelSize),
      content: drawingText,
      target: getExpectData(drawingText),
    } as {data: number[][], content: string; target: number[]};
    return newDataRecord;
  };

  const reshapeArray = (arr, size) => {
    const result = [];
    while (arr.length) {
      result.push(arr.splice(0, size));
    }
    return result;
  };

  const generateData = () => {
    const data = [...dataRecords, recordData()];
    console.info(data);
    setDataRecords(data);
    localStorage.setItem('tempData', JSON.stringify(data));
  };

  const downloadData = () => {
    const data = localStorage.getItem('tempData');
    if (data) {
      const blob = new Blob([data], { type: 'application/json' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'data.json';
      link.click();
    }
  };

  const handleCanvasWidthChange = event => {
    const width = parseInt(event.target.value, 10) || 1;
    setCanvasSize(prev => ({ ...prev, width }));
  };

  const handleCanvasHeightChange = event => {
    const height = parseInt(event.target.value, 10) || 1;
    setCanvasSize(prev => ({ ...prev, height }));
  };
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const generateTestData = () => {
    const testData = recordData();
    console.info('testData', testData);
    if (props?.onTestData) {
      props.onTestData([testData]);
    }
  };

  return (
    <div>
      <div>
        <label>
          Canvas Width:
          <input
            type="number"
            value={canvasSize.width}
            onChange={handleCanvasWidthChange}
          />
        </label>
        <label>
          Canvas Height:
          <input
            type="number"
            value={canvasSize.height}
            onChange={handleCanvasHeightChange}
          />
        </label>
      </div>
      <canvas
        ref={canvasRef}
        width={canvasSize.width * pixelSize}
        height={canvasSize.height * pixelSize}
        style={{ border: '1px solid #000' }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={finishDrawing}
        onMouseOut={finishDrawing}
      />
      <br />
      <label>
        Drawing Text:
        <input
          type="text"
          value={drawingText}
          onChange={handleDrawingTextInput}
        />
      </label>
      <br />
      <button onClick={generateData}>Generate Data</button>
      <button onClick={downloadData}>Download Data</button>
      <br />
      <button onClick={clearCanvas}>Clear Canvas</button>
      <button onClick={generateTestData}>Generate Test Data</button>
    </div>
  );
};

export default DrawingBoard;
