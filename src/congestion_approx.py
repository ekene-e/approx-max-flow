class CongestionApprox:
    # A congestion approximator represents (abstractly) a matrix R such that:
    # ||Rb||_inf <= opt(b) <= alpha ||Rb||_inf
    # IE, it hits node demand vectors, b, and gives an estimation of the congestion
    # incurred along some subset of edges within some factor alpha.

    def compute_dot(self, x):
        # For a congestion approximator R, compute R x
        # x should be a vector in node-space according to the order of nodes returned
        # by g.nodes()
        #
        # The result is a vector in some subset of the edge-space of g, with arbitrary
        # ordering except that the order should be consistent with the input of
        # compute_transpose_dot
        return None

    def compute_transpose_dot(self, x):
        # For a congestion approximator R, compute R^T x
        # The input is a vector in the same subset edge-space of the output of
        # compute_dot, and the output is a vector in the node-space of the graph
        return None

    def alpha(self):
        # For a congestion approximator R and demands b, return the error term alpha
        # In using ||RB||_inf to approximate the flow min congestion
        return None
