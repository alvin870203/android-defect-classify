// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.defectclassification;

import android.graphics.Rect;

import java.util.ArrayList;

class Result {
    int classIndex;
    Float score;
    Float roundProb;
    Float sharpProb;

    public Result(int cls, Float score, Float round, Float sharp) {
        this.classIndex = cls;
        this.score = score;
        this.roundProb = round;
        this.sharpProb = sharp;
    }
};

public class PrePostProcessor {
    // for net-0327_0135-wrap model, no need to apply MEAN and STD
    // the wrapped model handle Normalization internally using transforms
    public final static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    public final static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // wrapped model input image size (larger then model Net input image size)
    public final static int INPUT_WIDTH = 1080;
    public final static int INPUT_HEIGHT = 1080;
    public final static int OUTPUT_COLUMN = 2; // round and sharp

    static String[] mClasses;

    static ArrayList<Result> outputsToPredictions(float[] outputs) {
        ArrayList<Result> results = new ArrayList<>();
        int cls = (outputs[0] > outputs[1] ? 0 : 1);
        Result result = new Result(cls, outputs[cls], outputs[0], outputs[0]);
        results.add(result);
        return results;
    }
}
