import { useEffect, useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

const numPixels = 28
const initialPixels = new Array(numPixels).fill(new Array(numPixels).fill(0))

const canvasSize = 700

const xPixSize = canvasSize / numPixels
const yPixSize = canvasSize / numPixels

const paintRadius = 1

function App() {

  const [pixels, setPixels] = useState(initialPixels)
  const [isDrawing, setIsDrawing] = useState(false)
  const canvasRef = useRef("canvasRef")

  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    draw(e)
  }

  const draw = (e) => {
    const x = e.clientX - e.target.offsetLeft
    const y = e.clientY - e.target.offsetTop

    const xPix = parseInt(x / xPixSize)
    const yPix = parseInt(y / yPixSize)

    const result = pixels.map((row, rowIndex) => {
      return row.map((value, colIndex) => {
        
        // if (rowIndex == yPix && colIndex == xPix){
        //   return Math.min(1, value + 0.3)
        // }

        // if (Math.abs(rowIndex - yPix)**2 + Math.abs(colIndex - xPix)**2 <= paintRadius){
        //   const xDist = Math.abs(x - (colIndex * xPixSize + xPixSize / 2)) 
        //   const yDist = Math.abs(y - (rowIndex * yPixSize + yPixSize / 2)) 
          
        //   const dist = Math.sqrt(xDist**2 + yDist**2)

        //   return Math.min(value + Math.abs(dist - xPixSize) / xPixSize, 1)
        // }

        if (rowIndex-yPix <= 1 && rowIndex - yPix >= 0 && colIndex-xPix <= 1 && colIndex - xPix >= 0){
          return 1
        }

        return value

      })
    })

    setPixels(result)
  }

  useEffect(()=>{

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    pixels.map((row, rowIndex) => {
      row.map((value, colIndex) => {

        const color = 255 * value
        ctx.fillStyle = `rgb(${color}, ${color}, ${color})`; 
        ctx.fillRect(colIndex * xPixSize, rowIndex * yPixSize, xPixSize, yPixSize)
        

      })
    })
    

  }, [pixels])

  const sendImage = async() => {
    const res = await fetch("http://127.0.0.1:3838/predict", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(pixels)
      }
    )
    console.log(await res.json())
  }

  return (
    <>
      <canvas ref={canvasRef} style={{background: "black"}} onMouseMove={handleCanvasClick} onMouseUp={()=>setIsDrawing(false)} onMouseLeave={()=>setIsDrawing(false)} onMouseDown={(e)=>{setIsDrawing(true); draw(e)}} height={canvasSize} width={canvasSize}/>
      <div style={{display: "flex", flexDirection: "row"}}>
        <button onClick={()=>setPixels(initialPixels)}>Reset</button>
        <button onClick={sendImage}>Predict</button>
        <button onClick={()=>console.log(pixels)}>Get Drawing</button>
      </div>
    </>
  )
}

export default App
