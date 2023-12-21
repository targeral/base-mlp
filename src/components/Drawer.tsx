// DrawingBoard.js
import React, { useRef, useEffect } from 'react';

const DrawingBoard = () => {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineWidth = 5;
    ctxRef.current = ctx;
  }, []);

  const handleTouchMove = e => {
    const ctx = ctxRef.current;
    const pressure = e.touches[0].force || 0.5; // 获取触摸力度，如果不支持则默认为0.5
    console.info(pressure);
    // 根据力度计算灰度值
    const grayValue = Math.floor(pressure * 255);
    console.info(grayValue);

    ctx.strokeStyle = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
    ctx.lineTo(e.touches[0].clientX, e.touches[0].clientY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.touches[0].clientX, e.touches[0].clientY);
  };

  return (
    <div style={{ border: '1px solid #ccc' }}>
      可以识别触摸板的压力画板：
      <canvas
        ref={canvasRef}
        onTouchMove={handleTouchMove}
        onTouchStart={e => {
          console.info('aaa');
          const ctx = ctxRef.current;
          ctx.beginPath();
          ctx.moveTo(e.touches[0].clientX, e.touches[0].clientY);
        }}
      />
    </div>
  );
};

export default DrawingBoard;
