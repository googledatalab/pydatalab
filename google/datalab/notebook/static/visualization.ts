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

// require.js plugin to allow Google Chart API to be loaded.

/// <reference path="../../../externs/ts/require/require.d.ts" />

declare var google: any;
declare var window: any;

module Visualization {

  'use strict';

  // Queued packages to load until the google api loader itself has not been loaded.
  var queue: any = {
    packages: [],
    callbacks: []
  };

  function loadGoogleApiLoader(callback: any): void {
    // Visualization packages are loaded using the Google loader.
    // The loader URL itself must contain a callback (by name) that it invokes when its loaded.
    var callbackName: string = '__googleApiLoaderCallback';
    window[callbackName] = callback;

    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.async = true;
    script.src = 'https://www.google.com/jsapi?callback=' + callbackName;
    document.getElementsByTagName('head')[0].appendChild(script);
  }

  function invokeVisualizationCallback(cb: any) {
    cb(google.visualization);
  }

  function loadVisualizationPackages(names: any, callbacks: any): void {
    if (names.length) {
      var visualizationOptions = {
        packages: names,
        callback: function() { callbacks.forEach(invokeVisualizationCallback); }
      };

      google.load('visualization', '1', visualizationOptions);
    }
  }

  loadGoogleApiLoader(function() {
    if (queue) {
      loadVisualizationPackages(queue.packages, queue.callbacks);
      queue = null;
    }
  });

  export function load(name: any, req: any, callback: any, config: any) {
    if (config.isBuild) {
      callback(null);
    }
    else {
      if (queue) {
        // Queue the package and associated callback to load, once the loader has been loaded.
        queue.packages.push(name);
        queue.callbacks.push(callback);
      }
      else {
        // Loader has already been loaded, so go ahead and load the specified package.
        loadVisualizationPackages([ name ], [ callback ]);
      }
    }
  }
}

export = Visualization;
