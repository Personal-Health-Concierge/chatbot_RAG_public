
import React from "react";
import { useState } from "react";
import { Line } from "react-chartjs-2";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faAngleRight, faAngleDown } from '@fortawesome/free-solid-svg-icons'


/*<h4 style={{ textAlign: "center" }} onClick={onClick}>{chartTitle}</h4>*/

export default function LineChart({ title, data }) {

  const [show, setShow] = useState(false)
  const onClick = () => setShow(!show)

  let dates = []
  let datasets = []

  if (data && data.length > 0) {
    // Extract dates and keys
    dates = data.map(entry => entry.date);
    const keys = Object.keys(data[0]).filter(key => key !== 'date');
    const colors = {
      'total_cholesterol': 'green',
      'triglycerides': 'red',
      'hdl_cholesterol': 'blue',
      'ldl_cholesterol': 'orange'
    }

    // Generate datasets dynamically
    datasets = keys.map((key, index) => ({
      label: key.replace('_', ' ').toUpperCase(),
      data: data.map(entry => entry[key]),
      /* backgroundColor: colors[index % colors.length].replace('1)', '0.2)'), */
      borderColor: colors[key],
      borderWidth: 2 // Adjust line thickness
    }));

    return (
      <div
        style={{
          margin: "0.5rem",
          padding: "0.5rem",
          borderTop: "solid 1pt",
          borderColor: "#D3D3D3"
        }}
      >
        <div style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
        }}>
          <FontAwesomeIcon icon={show ? faAngleDown : faAngleRight}
            onClick={onClick} />
          <div style={{
            marginLeft: "auto",
            marginRight: "auto",
            fontWeight: "600"
          }}>{title}</div>
        </div>
        {show ?
          <Line
            style={{
              maxHeight: "300pt",
              overflowY: "scroll"
            }}
            data={{
              labels: dates,
              datasets: datasets
            }}
            options={{
              maintainAspectRatio: false,
              scales: {
                y: {
                  beginAtZero: true
                }
              },
              plugins: {
                legend: {
                  position: 'bottom', // Position the legend at the top
                },
                title: {
                  display: false,
                  text: title,
                  font: {
                    size: 24 // Change the font size
                  }
                }
              }
            }} /> : null}
      </div>
    );
  } else
    return (<></>)
}



/*
  // Extract dates and keys
  const dates = data.map(entry => entry.date);
  const keys = Object.keys(data[0]).filter(key => key !== 'date');

  // Function to generate dynamic colors based on the number of datasets
  function generateColors(numColors, colorLength) {
    const colors = [];
    for (let i = 0; i < numColors; i++) {
        const color = getRandomColor(colorLength);
        colors.push(color);
    }
    return colors;
  }

  function getRandomColor(length) {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < length; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  // Generate colors based on the number of datasets
  const colors = generateColors(keys.length, keys.length);

  // Generate datasets dynamically
  const datasets = keys.map((key, index) => ({
    label: key.replace('_', ' ').toUpperCase(),
    data: data.map(entry => entry[key]),
    backgroundColor: colors[index % colors.length].replace('1)', '0.2)'),
    borderColor: colors[index % colors.length],
    borderWidth: 3 // Adjust line thickness
  }));

  // Chart.js configuration
  const ctx = document.getElementById('PCoH').getContext('2d');
  const myChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: dates,
          datasets: datasets
      },
      options: {
          scales: {
              y: {
                  beginAtZero: true
              }
          },
          plugins: {
              legend: {
                  position: 'bottom', // Position the legend at the top
              },
              title: {
                  display: true,
                  text: 'Cholesterol Levels Over Time',
                  font: {
                      size: 24 // Change the font size
                  }
              }
          }
      }
  });
  */