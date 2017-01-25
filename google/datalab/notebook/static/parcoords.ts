/*
 * Copyright 2016 Google Inc. All rights reserved.
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

/// <reference path="../../../externs/ts/require/require.d.ts" />

module ParCoords {

  function getCentroids(data: any, graph: any): any {
      var margins = graph.margin();
      var graphCentPts: any[] = [];

      data.forEach(function(d: any){
        var initCenPts = graph.compute_centroids(d).filter(function(d: any, i: number) {return i%2==0;});
        var cenPts = initCenPts.map(function(d: any){
        return [d[0] + margins["left"], d[1]+ margins["top"]]; 
      });
      graphCentPts.push(cenPts);
    });

    return graphCentPts;
  }

  function getActiveData(graph: any): any{
    if (graph.brushed()!=false) return graph.brushed();
      return graph.data();
  }

  function findAxes(testPt: any, cenPts: any): number {
    var x: number = testPt[0];
    var y: number = testPt[1];
    if (cenPts[0][0] > x) return 0;
    if (cenPts[cenPts.length-1][0] < x) return 0;
    for (var i=0; i<cenPts.length; i++) {
        if (cenPts[i][0] > x) return i;
    }
    return 0;
  }

  function isOnLine(startPt: any, endPt: any, testPt: any, tol: number){
    var x0 = testPt[0];
    var	y0 = testPt[1];
    var x1 = startPt[0];
    var	y1 = startPt[1];
    var x2 = endPt[0];
    var y2 = endPt[1];
    var Dx = x2 - x1;
    var Dy = y2 - y1;
    var delta = Math.abs(Dy*x0 - Dx*y0 - x1*y2+x2*y1)/Math.sqrt(Math.pow(Dx, 2) + Math.pow(Dy, 2)); 
    if (delta <= tol) return true;
    return false;
  }

  function getClickedLines(mouseClick: any, graph: any): any {
    var clicked: any[] = [];
    var clickedCenPts: any[] = [];
    // find which data is activated right now
    var activeData: any = getActiveData(graph);

    // find centriod points
    var graphCentPts: any = getCentroids(activeData, graph);

    if (graphCentPts.length==0) return false;
    // find between which axes the point is
    var axeNum: number = findAxes(mouseClick, graphCentPts[0]);
    if (!axeNum) return false;
    graphCentPts.forEach(function(d: any, i: number){
      if (isOnLine(d[axeNum-1], d[axeNum], mouseClick, 2)) {
        clicked.push(activeData[i]);
        clickedCenPts.push(graphCentPts[i]); // for tooltip
      }
    });
    return [clicked, clickedCenPts]
  }

  function highlightLineOnClick(mouseClick: any, graph: any) {
    var clicked: any[] = [];
    var clickedCenPts: any[] = [];
    var clickedData: any = getClickedLines(mouseClick, graph);
    if (clickedData && clickedData[0].length!=0){
      clicked = clickedData[0];
      clickedCenPts = clickedData[1];
      // highlight clicked line
      graph.highlight(clicked);
    }
  };

  export function plot(d3: any, color_domain: number[], maximize: boolean, data: any,
           graph_html_id: string, grid_html_id: string) {
    var range = ["green", "gray"];
    if (maximize) {
      range = ["gray", "green"];
    }
    var blue_to_brown = d3.scale.linear().domain(color_domain)
                                         .range(range)
                                         .interpolate(d3.interpolateLab);
    var color = function(d: any) { return blue_to_brown(d['Objective']); };
    var columns_hide: string[] = ["Trial", "Training Step"];
    for (var attr in data) {
      if (attr.lastIndexOf("(log)") > 0) {
        columns_hide.push(attr.slice(0, -5));
      }
    }
    var data_display: any[] = [];
    for (var i: number =0; i<data.Trial.length; i++) {
      var instance: any = {};
      for (var attr in data) {
        instance[attr] = data[attr][i];
      }
      data_display.push(instance);
    }
    var parcoords = d3.parcoords()("#" + graph_html_id).color(color).alpha(0.4);
    parcoords.data(data_display).hideAxis(columns_hide)
                                .composite("darken")
                                .render()
                                .brushMode("1D-axes");

    var grid = d3.divgrid();
    d3.select("#" + grid_html_id).datum(data_display)
        .call(grid)
        .selectAll(".row")
        .on({
          "mouseover": function(d: any) { parcoords.highlight([d]) },
          "mouseout": parcoords.unhighlight
        });     
    // update data table on brush event
    parcoords.on("brush", function(d: any) {
        d3.select("#" + grid_html_id).datum(d)
        .call(grid)
        .selectAll(".row")
        .on({
            "mouseover": function(d: any) { parcoords.highlight([d]) },
            "mouseout": parcoords.unhighlight
        });
    });
    //add hover event
    d3.select("#" + graph_html_id + " svg").on("mousemove", function() {
          var mousePosition: any = d3.mouse(this);			    
          highlightLineOnClick(mousePosition, parcoords);
        })
        .on("mouseout", function(){
          parcoords.unhighlight();
	});
  }
}


export = ParCoords;

