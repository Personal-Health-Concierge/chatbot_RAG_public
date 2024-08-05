// App.js
import Chart from "chart.js/auto";
import { CategoryScale } from "chart.js";
import { useState } from "react";
import { Data } from "./Data";
import PieChart from "../components/PieChart";
import BarChart from "../components/BarChart";
import LineChart from "../components/LineChart";
import "./styles.css";

Chart.register(CategoryScale);

export default function App() {
    const [chartData, setChartData] = useState({
        // ...chart data
    });

    return (
        <div className="App">
            <PieChart chartData={chartData} />
            <BarChart chartData={chartData} />
            <LineChart chartData={chartData} />
        </div>
    );
}
