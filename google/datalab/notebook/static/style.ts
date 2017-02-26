/*
 * Copyright 2015 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

// RequireJS plugin to load stylesheets.

/// <reference path="../../../externs/ts/require/require.d.ts" />

module Style {

  'use strict';

  // An object containing the set of loaded stylesheets, so as to avoid reloading.
  var loadedStyleSheets: any = {};

  // An object containing stylesheets to load, once the DOM is ready.
  var pendingStyleSheets: Array<string> = null;

  function addStyleSheet(url: string): void {
    loadedStyleSheets[url] = true;

    var stylesheet = document.createElement('link');
    stylesheet.type = 'text/css';
    stylesheet.rel = 'stylesheet';
    stylesheet.href = url;

    document.getElementsByTagName('head')[0].appendChild(stylesheet);
  }

  function domReadyCallback(): void {
    if (pendingStyleSheets) {
      // Clear out pendingStyleSheets, so any future adds are immediately processed.
      var styleSheets: Array<string> = pendingStyleSheets;
      pendingStyleSheets = null;

      styleSheets.forEach(addStyleSheet);
    }
  }

  export function load(url: string, req: any, loadCallback: any, config: any): void {
    if (config.isBuild) {
      loadCallback(null);
    }
    else {
      // Go ahead and immediately/optimistically resolve this, since the resolved value of a
      // stylesheet is never interesting.
      setTimeout(loadCallback, 0);

      // Only load a specified stylesheet once for the lifetime of this page.
      if (loadedStyleSheets[url]) {
        return;
      }
      loadedStyleSheets[url] = true;

      if (document.readyState == 'loading') {
        if (!pendingStyleSheets) {
          pendingStyleSheets = [];
          document.addEventListener('DOMContentLoaded', domReadyCallback, false);
        }

        pendingStyleSheets.push(url);
      }
      else {
        addStyleSheet(url);
      }
    }
  }
}

export = Style;
