import numpy as np

def get_pacing_function(pacing_f, a,b, total_step, total_data):
        """Return a  pacing function  w.r.t. step.
        input:
        a:[1,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step,
        total_data)) 
        b:[1,1]  percentatge of total data at the begining of the training. Thia is a starting point (1,b*total_data))
        """
         
        index_start = b*total_data
        if pacing_f == 'linear':
            rate = (total_data - index_start)/(a*total_step)
            def _linear_function(step):
                return np.clip(int(rate *step + index_start), 1,total_data);
            return _linear_function

        elif pacing_f == 'quad':
            rate = (total_data-index_start)/(a*total_step)**2  
            def _quad_function(step):
                return np.clip(int(rate*step**2 + index_start), 1,total_data);
            return _quad_function

        elif pacing_f == 'sqrt':
            rate = (total_data-index_start)/(a*total_step)**0.5
            def _root_function(step):
                return np.clip(int(rate *step**0.5 + index_start), 1,total_data);
            return _root_function

        elif pacing_f == 'step':
            threshold = a*total_step
            def _step_function(step):
                return np.clip(int( total_data*(step//threshold) +index_start), 1,total_data);
            return _step_function      

        elif pacing_f == 'exp':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            rate =  (total_data-tilde_b)/(np.exp(c)-1)
            constant = c/tilde_a
            def _exp_function(step):
                if not np.isinf(np.exp(step *constant)):
                    return np.clip(int(rate*(np.exp(step*constant)-1) + tilde_b), 1,total_data);
                else:
                    return total_data
            return _exp_function

        elif pacing_f == 'log':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            ec = np.exp(-c)
            N_b = (total_data-tilde_b)
            def _log_function(step):
                return np.clip(int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b), 1,total_data);
            return _log_function