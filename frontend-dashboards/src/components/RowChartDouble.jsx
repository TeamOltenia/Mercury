import React, {useEffect, useRef, useState} from 'react';
import Chart from 'chart.js/auto'; // Importing Chart.js

const ChartComponent2 = () => {
  const chartRef2 = useRef(null);
  const [chart, setChart] = useState(null); // State to keep track of the chart instance

  const chartInstance2 = useRef(null);  // Using ref to store the chart instance directly

  useEffect(() => {
    async function fetchChartData() {
      console.log("Fetching additional data...");
      const response = await fetch('http://127.0.0.1:8000/data/C38997010');
      const rawData = await response.json();

      const steps = rawData.map(txn => txn.step);
      const oldBalance = rawData.map(txn => txn.oldbalanceOrg);
      const newBalance = rawData.map(txn => txn.newbalanceOrig);

      return {
        labels: steps,
        datasets: [
          {
            label: 'Old Balance',
            data: oldBalance,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
          {
            label: 'New Balance',
            data: newBalance,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
          }
        ]
      };
    }

    const initializeChart = async () => {
      const { labels, datasets } = await fetchChartData();

      const chartConfig = {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive: true,
          plugins: {
            legend: { position: 'top' },
          }
        }
      };

      if (chartRef2.current) {
        if (chart) {
          chart.destroy(); // Ensure the existing chart is destroyed
        }
        const newChart = new Chart(chartRef2.current, chartConfig);
        setChart(newChart);
      }
    };

    initializeChart();

    // Cleanup function to destroy the chart when the component unmounts
    return () => {
      if (chart) {
        chart.destroy();
      }
    };
  }, []); // Empty dependency array ensures this effect runs only once on mount

  return (

      <div style={{width: '100vh', height: '60vh', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <canvas ref={chartRef2} style={{width: '100%', height: 'auto'}}/>
      </div>
  );
};

export default ChartComponent2;
