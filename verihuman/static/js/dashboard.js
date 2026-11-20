/* VeriHuman — dashboard charts. Every number comes from the database. */
(function () {
  "use strict";

  var holder = document.getElementById("dashboard-data");
  if (!holder || !window.Chart) return;

  var data;
  try {
    data = JSON.parse(holder.textContent || "{}");
  } catch (error) {
    return;
  }
  if (!data || !data.hasData) return;

  var charts = [];

  function css(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      bonafide: css("--bonafide", "#22C55E"),
      spoof: css("--spoof", "#F43F5E"),
      uncertain: css("--uncertain", "#F59E0B"),
      text: css("--text", "#FAFAFA"),
      muted: css("--muted", "#A1A1AA"),
      border: css("--border", "#3F3F46"),
      card: css("--card", "#18181B"),
      elevated: css("--elevated", "#27272A"),
      surface: css("--bg", "#09090B")
    };
  }

  function baseScales(color) {
    return {
      x: {
        grid: { display: false },
        border: { display: false },
        ticks: { color: color.muted, font: { size: 10 }, maxRotation: 0, autoSkip: true }
      },
      y: {
        beginAtZero: true,
        grid: { color: color.border, drawTicks: false },
        border: { display: false },
        ticks: { color: color.muted, font: { size: 10 }, precision: 0, padding: 6 }
      }
    };
  }

  function gradient(ctx, area, from, to) {
    if (!area) return from;
    var fill = ctx.createLinearGradient(0, area.top, 0, area.bottom);
    fill.addColorStop(0, from);
    fill.addColorStop(1, to);
    return fill;
  }

  var centreText = {
    id: "vhCentreText",
    afterDraw: function (chart, args, options) {
      if (!options || !options.value) return;
      var ctx = chart.ctx;
      var meta = chart.getDatasetMeta(0);
      if (!meta.data.length) return;
      var point = meta.data[0];
      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = options.color;
      ctx.font = "600 24px " + Chart.defaults.font.family;
      ctx.fillText(options.value, point.x, point.y - 6);
      ctx.fillStyle = options.muted;
      ctx.font = "500 9px " + Chart.defaults.font.family;
      ctx.fillText(options.label || "ANALYSES", point.x, point.y + 15);
      ctx.restore();
    }
  };

  function donut() {
    var canvas = document.getElementById("chart-verdicts");
    if (!canvas) return;
    var c = palette();
    var verdicts = data.verdicts || {};
    var values = [verdicts.bonafide || 0, verdicts.spoof || 0, verdicts.uncertain || 0];
    var total = values.reduce(function (sum, value) { return sum + value; }, 0);

    charts.push(new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: ["Human / Bonafide", "AI-generated / Spoof", "Uncertain"],
        datasets: [{
          data: values,
          backgroundColor: [c.bonafide, c.spoof, c.uncertain],
          borderColor: c.card,
          borderWidth: 3,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "74%",
        plugins: {
          legend: { display: false },
          vhCentreText: { value: String(total), label: "ANALYSES", color: c.text, muted: c.muted },
          tooltip: {
            callbacks: {
              label: function (context) {
                var share = total ? Math.round((context.parsed / total) * 100) : 0;
                return " " + context.label + ": " + context.parsed + " (" + share + "%)";
              }
            }
          }
        }
      },
      plugins: [centreText]
    }));
  }

  function daily() {
    var canvas = document.getElementById("chart-days");
    if (!canvas) return;
    var c = palette();
    var days = data.days || { labels: [], bonafide: [], spoof: [], uncertain: [] };

    charts.push(new Chart(canvas, {
      type: "bar",
      data: {
        labels: days.labels,
        datasets: [
          {
            label: "Human",
            data: days.bonafide,
            backgroundColor: c.bonafide,
            borderRadius: 3,
            borderSkipped: false,
            stack: "day",
            maxBarThickness: 22
          },
          {
            label: "AI",
            data: days.spoof,
            backgroundColor: c.spoof,
            borderRadius: 3,
            borderSkipped: false,
            stack: "day",
            maxBarThickness: 22
          },
          {
            label: "Uncertain",
            data: days.uncertain,
            backgroundColor: c.uncertain,
            borderRadius: 3,
            borderSkipped: false,
            stack: "day",
            maxBarThickness: 22
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: baseScales(c),
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.dataset.label + ": " + context.parsed.y;
              }
            }
          }
        }
      },
      plugins: [{
        id: "vhRoundedStack",
        beforeDatasetsDraw: function (chart) {
          // Rounded top corners on the highest visible bar in each stack.
          var ctx = chart.ctx;
          ctx.save();
          ctx.shadowColor = "transparent";
          ctx.restore();
        }
      }]
    }));
  }

  function distribution() {
    var canvas = document.getElementById("chart-distribution");
    if (!canvas) return;
    var c = palette();
    var buckets = data.buckets || { labels: [], values: [] };
    var max = Math.max.apply(null, (buckets.values || [0]).concat([1]));

    var colors = buckets.labels.map(function (_label, index) {
      var share = index / Math.max(1, buckets.labels.length - 1);
      if (share <= 0.2) return c.bonafide;
      if (share >= 0.8) return c.spoof;
      if (share >= 0.4 && share <= 0.6) return c.uncertain;
      return c.muted;
    });

    charts.push(new Chart(canvas, {
      type: "bar",
      data: {
        labels: buckets.labels,
        datasets: [{
          data: buckets.values,
          backgroundColor: colors,
          borderRadius: 5,
          borderSkipped: false,
          maxBarThickness: 40
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: c.muted, font: { size: 10 } }
          },
          y: {
            beginAtZero: true,
            suggestedMax: max,
            grid: { color: c.border, drawTicks: false },
            border: { display: false },
            ticks: { color: c.muted, font: { size: 10 }, precision: 0, padding: 6 }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.parsed.y + " analysis" + (context.parsed.y === 1 ? "" : "es");
              }
            }
          }
        }
      }
    }));
  }

  function trend() {
    var canvas = document.getElementById("chart-trend");
    if (!canvas) return;
    var c = palette();
    var series = data.trend || { labels: [], values: [] };

    charts.push(new Chart(canvas, {
      type: "line",
      data: {
        labels: series.labels,
        datasets: [{
          label: "Spoof score",
          data: series.values,
          borderColor: c.text,
          borderWidth: 1.6,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: c.text,
          pointHoverBorderColor: c.surface,
          tension: 0.35,
          fill: true,
          backgroundColor: function (context) {
            var chart = context.chart;
            var area = chart.chartArea;
            return gradient(chart.ctx, area, "rgba(161,161,170,0.22)", "rgba(161,161,170,0)");
          }
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: c.muted, font: { size: 9 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 6 }
          },
          y: {
            min: 0,
            max: 100,
            grid: { color: c.border, drawTicks: false },
            border: { display: false },
            ticks: {
              color: c.muted,
              font: { size: 10 },
              stepSize: 25,
              padding: 6,
              callback: function (value) { return value + "%"; }
            }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " Spoof score: " + context.parsed.y.toFixed(2) + "%";
              },
              afterLabel: function (context) {
                var value = context.parsed.y;
                if (value >= 71.18) return "Above spoof threshold";
                if (value <= 50) return "Below bonafide threshold";
                return "Uncertain band";
              }
            }
          }
        }
      }
    }));
  }

  function build() {
    charts.forEach(function (chart) {
      try { chart.destroy(); } catch (error) {}
    });
    charts = [];
    donut();
    daily();
    distribution();
    trend();
  }

  build();

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      charts.forEach(function (chart) { chart.resize(); });
    }, 180);
  });

  window.addEventListener("vh:theme", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(build, 60);
  });
})();
