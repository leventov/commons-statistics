package org.apache.commons.statistics.distribution;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class InverseBinomialCdfNormalApproximationBiasTest {

    private static final SplittableRandom REPEATABLE_RANDOM = new SplittableRandom(0);

    private static final double PROB_OF_SUCCESS = 0.9;

    private static final double N_EXP_BASE = 1.01;
    private static final int MIN_N = 100_000;
    private static final int MAX_N = 1_000_000_000;

    private static final double MIN_P = 0.8;
    private static final double MAX_P = 0.999;
    private static final int P_STEPS = 20;

    private static final double P_EXP_BASE = 1 / Math.pow((1 - MIN_P) / (1 - MAX_P), 1.0 / P_STEPS);

    public static void main(String[] args) {
        for (double p : new double[] {0.77, 0.9, 0.99}) {
            System.out.println("CDF parameter = " + p);
            String s = IntStream.range(MIN_N, MIN_N + 100)
                    .map(n -> inverseBinomialCdfNormalApproximationError(PROB_OF_SUCCESS, n, p))
                    .mapToObj(Integer::toString)
                    .collect(Collectors.joining(" "));
            System.out.println(s);
            String s2 = IntStream.range(MAX_N - 100, MAX_N)
                    .map(n -> inverseBinomialCdfNormalApproximationError(PROB_OF_SUCCESS, n, p))
                    .mapToObj(Integer::toString)
                    .collect(Collectors.joining(" "));
            System.out.println(s2);
        }
        for (double pComplement = 1.0 - MIN_P; pComplement >= 1.0 - MAX_P;
             pComplement *= P_EXP_BASE) {
            double p = 1.0 - pComplement;

            double meanError = estimateMeanError(p);

            System.out.println(p + " " + meanError);
        }
    }

    private static double estimateMeanError(double p) {
        List<Integer> ns = new ArrayList<>();
        for (int baseN = MIN_N; baseN < MAX_N; baseN *= N_EXP_BASE) {
            int nLimit = (int) (baseN * N_EXP_BASE);
            // Sample size of 1000 per "n log" ensures approximately 3-digit precision (+/- 0.001)
            // of MAE. That could be verified by removing seed from REPEATABLE_RANDOM and seeing how
            // this program prints different values from run to run.
            int nsSampleSize = 1000;
            ns.addAll(generateRandomSelection(nsSampleSize, baseN, nLimit));
        }
        return ns
                .stream()
                .parallel()
                .mapToInt(n -> inverseBinomialCdfNormalApproximationError(PROB_OF_SUCCESS, n, p))
                .average()
                .getAsDouble();
    }

    private static Collection<Integer> generateRandomSelection(
            int sampleSize, int origin, int bound) {
        if (bound - origin <= sampleSize) {
            return IntStream.range(origin, bound).boxed().collect(Collectors.toList());
        }
        Set<Integer> selected = new HashSet<>();
        while (selected.size() < sampleSize) {
            selected.add(REPEATABLE_RANDOM.nextInt(origin, bound));
        }
        return selected;
    }

    /**
     * @param probOfSuccess the parameter of BinomialDistribution
     * @param n the parameter of BinomialDistribution
     * @param p the parameter of CDF
     */
    private static int inverseBinomialCdfNormalApproximationError(
            double probOfSuccess, int n, double p) {
        int binomialResult = new BinomialDistribution(n, probOfSuccess)
                .inverseCumulativeProbability(p);
        int approxNormalResult = Math.toIntExact(Math.round(
                inverseBinomialCdfNormalApproximation(probOfSuccess, n, p)));
        return approxNormalResult - binomialResult;
    }


    /**
     * @param probOfSuccess the parameter of BinomialDistribution
     * @param n the parameter of BinomialDistribution
     * @param p the parameter of CDF
     */
    private static double inverseBinomialCdfNormalApproximation(
            double probOfSuccess, int n, double p) {
        double mean = n * probOfSuccess;
        double stdDev = Math.sqrt(n * probOfSuccess * (1 - probOfSuccess));
        return new NormalDistribution(mean, stdDev).inverseCumulativeProbability(p);
    }
}
