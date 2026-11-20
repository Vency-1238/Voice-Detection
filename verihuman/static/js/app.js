/* VeriHuman — shared UI behaviour (theme, navigation, menus, helpers). */
(function () {
  "use strict";

  var THEME_KEY = "vh-theme";

  var VH = (window.VH = window.VH || {});

  /* ------------------------------------------------------------- theme --- */

  function currentTheme() {
    var stored = null;
    try {
      stored = localStorage.getItem(THEME_KEY);
    } catch (error) {}
    if (stored === "light" || stored === "dark") return stored;
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function paintTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "light" ? "#FFFFFF" : "#09090B");
    window.dispatchEvent(new CustomEvent("vh:theme", { detail: { theme: theme } }));
  }

  function setTheme(theme, remember) {
    paintTheme(theme);
    if (remember !== false) {
      try {
        localStorage.setItem(THEME_KEY, theme);
      } catch (error) {}
    }
  }

  VH.theme = currentTheme;
  VH.setTheme = setTheme;

  paintTheme(currentTheme());

  /* ------------------------------------------------------------- icons --- */

  function drawIcons(root) {
    if (!window.lucide || typeof window.lucide.createIcons !== "function") return;
    try {
      window.lucide.createIcons({
        attrs: { "stroke-width": 1.75 },
        nameAttr: "data-lucide",
        root: root || document
      });
    } catch (error) {
      /* icons are decorative; never break the page */
    }
  }

  VH.icons = drawIcons;

  /* ------------------------------------------------------------- csrf ---- */

  VH.csrfToken = function () {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  };

  VH.apiFetch = function (url, options) {
    var opts = options || {};
    var headers = new Headers(opts.headers || {});
    var token = VH.csrfToken();
    if (token) headers.set("X-CSRFToken", token);
    headers.set("Accept", "application/json");
    opts.headers = headers;
    opts.credentials = "same-origin";
    return fetch(url, opts).then(function (response) {
      return response
        .json()
        .catch(function () {
          return { error: "The server returned an unreadable response." };
        })
        .then(function (payload) {
          if (!response.ok) {
            var message = (payload && payload.error) || "Request failed (" + response.status + ").";
            var error = new Error(message);
            error.status = response.status;
            error.payload = payload;
            throw error;
          }
          return payload;
        });
    });
  };

  /* ------------------------------------------------------------ helpers -- */

  VH.formatBytes = function (bytes) {
    var value = Number(bytes) || 0;
    if (value >= 1024 * 1024) return (value / (1024 * 1024)).toFixed(2) + " MB";
    if (value >= 1024) return (value / 1024).toFixed(1) + " KB";
    return Math.round(value) + " B";
  };

  VH.formatDuration = function (seconds) {
    var value = Number(seconds);
    if (!isFinite(value) || value < 0) return "—";
    var mins = Math.floor(value / 60);
    var rest = value - mins * 60;
    if (mins > 0) return mins + "m " + rest.toFixed(1).padStart(4, "0") + "s";
    return rest.toFixed(1) + "s";
  };

  VH.formatMs = function (ms) {
    var value = Number(ms);
    if (!isFinite(value)) return "—";
    if (value >= 1000) return (value / 1000).toFixed(2) + " s";
    return Math.round(value) + " ms";
  };

  VH.escape = function (value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
    });
  };

  VH.iconMarkup = function (name, size) {
    return (
      '<i data-lucide="' + name + '" class="vh-icon" style="width:' +
      (size || 18) + "px;height:" + (size || 18) + 'px" aria-hidden="true"></i>'
    );
  };

  /* ------------------------------------------------------------- header -- */

  var publicToggle = document.getElementById("public-nav-toggle");
  var publicLinks = document.getElementById("public-nav-links");
  if (publicToggle && publicLinks) {
    publicToggle.addEventListener("click", function () {
      var open = publicLinks.classList.toggle("open");
      publicToggle.setAttribute("aria-expanded", open ? "true" : "false");
      publicToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    });
  }

  var sidebarActive = false;
  function closeSidebar() {
    if (!sidebarActive) return;
    sidebarActive = false;
    document.body.classList.remove("sidebar-open");
    var btn = document.getElementById("sidebar-toggle");
    if (btn) btn.setAttribute("aria-expanded", "false");
  }
  function openSidebar() {
    sidebarActive = true;
    document.body.classList.add("sidebar-open");
    var btn = document.getElementById("sidebar-toggle");
    if (btn) btn.setAttribute("aria-expanded", "true");
  }

  var sidebarToggle = document.getElementById("sidebar-toggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
      sidebarActive ? closeSidebar() : openSidebar();
    });
  }
  var sidebarClose = document.getElementById("sidebar-close");
  if (sidebarClose) sidebarClose.addEventListener("click", closeSidebar);
  document.addEventListener("click", function (event) {
    if (!sidebarActive) return;
    var sidebar = document.querySelector(".vh-sidebar");
    if (sidebar && !sidebar.contains(event.target) && event.target !== sidebarToggle) {
      closeSidebar();
    }
  });

  document.querySelectorAll(".vh-nav-links a").forEach(function (link) {
    link.addEventListener("click", function () {
      if (publicLinks) publicLinks.classList.remove("open");
    });
  });

  /* ----------------------------------------------------------- dropdowns -- */

  function closeMenus(except) {
    document.querySelectorAll("[data-dropdown]").forEach(function (panel) {
      if (panel !== except) panel.hidden = true;
    });
    document.querySelectorAll("[data-dropdown-trigger]").forEach(function (trigger) {
      if (!except || trigger.getAttribute("aria-controls") !== except.id) {
        trigger.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.querySelectorAll("[data-dropdown-trigger]").forEach(function (trigger) {
    trigger.addEventListener("click", function (event) {
      event.stopPropagation();
      var panel = document.getElementById(trigger.getAttribute("aria-controls"));
      if (!panel) return;
      var willOpen = panel.hidden;
      closeMenus(willOpen ? panel : null);
      panel.hidden = !willOpen;
      trigger.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });
  });

  document.addEventListener("click", function (event) {
    if (!event.target.closest("[data-dropdown]")) closeMenus(null);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeMenus(null);
      closeSidebar();
    }
  });

  /* ------------------------------------------------------------ flashes -- */

  document.querySelectorAll("[data-dismiss]").forEach(function (button) {
    button.addEventListener("click", function () {
      var alert = button.closest(".vh-alert");
      if (!alert) return;
      alert.style.transition = "opacity .2s ease, transform .2s ease";
      alert.style.opacity = "0";
      alert.style.transform = "translateY(-6px)";
      setTimeout(function () {
        alert.remove();
      }, 200);
    });
  });

  document.querySelectorAll(".vh-alert").forEach(function (alert, index) {
    setTimeout(function () {
      if (!document.body.contains(alert)) return;
      alert.style.transition = "opacity .3s ease";
      alert.style.opacity = "0";
      setTimeout(function () {
        alert.remove();
      }, 320);
    }, 7000 + index * 400);
  });

  /* ------------------------------------------------------------ confirms -- */

  document.querySelectorAll("[data-confirm]").forEach(function (element) {
    element.addEventListener("click", function (event) {
      if (!window.confirm(element.getAttribute("data-confirm"))) {
        event.preventDefault();
        event.stopPropagation();
      }
    });
  });

  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (!window.confirm(form.getAttribute("data-confirm"))) event.preventDefault();
    });
  });

  /* ------------------------------------------------------------- theme UI -- */

  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    button.addEventListener("click", function () {
      setTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  });

  // Settings page radios change the theme immediately, and the form still saves it.
  document.querySelectorAll('input[name="theme"]').forEach(function (radio) {
    radio.addEventListener("change", function () {
      if (radio.checked) setTheme(radio.value);
    });
  });

  /* ---------------------------------------------------- password visibility */

  document.querySelectorAll("[data-toggle-password]").forEach(function (button) {
    button.addEventListener("click", function () {
      var input = document.getElementById(button.getAttribute("data-toggle-password"));
      if (!input) return;
      var show = input.type === "password";
      input.type = show ? "text" : "password";
      button.setAttribute("aria-label", show ? "Hide password" : "Show password");
      button.innerHTML = VH.iconMarkup(show ? "eye-off" : "eye", 16);
      drawIcons(button);
    });
  });

  /* ------------------------------------------------------------- copy ----- */

  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", function () {
      var value = button.getAttribute("data-copy");
      if (!value) return;
      var done = function () {
        var previous = button.getAttribute("title") || "Copy";
        button.setAttribute("title", "Copied");
        button.classList.add("copied");
        setTimeout(function () {
          button.setAttribute("title", previous);
          button.classList.remove("copied");
        }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(value).then(done).catch(function () {});
      }
    });
  });

  /* ------------------------------------------------------------ charts ----- */

  if (window.Chart) {
    Chart.defaults.font.family =
      getComputedStyle(document.documentElement).getPropertyValue("--font").trim() ||
      "Inter, system-ui, sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.animation.duration = 700;
    Chart.defaults.plugins.legend.display = false;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(9,9,11,0.94)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(250,250,250,0.14)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleFont = { size: 11, weight: "600" };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 11 };
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.displayColors = true;
    Chart.defaults.plugins.tooltip.boxWidth = 8;
    Chart.defaults.plugins.tooltip.boxHeight = 8;
  }

  /* ------------------------------------------------------------ startup --- */

  drawIcons(document);
})();
