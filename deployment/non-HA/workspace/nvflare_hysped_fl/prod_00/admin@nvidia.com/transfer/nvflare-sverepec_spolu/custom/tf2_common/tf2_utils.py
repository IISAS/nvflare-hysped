from tf2_common.tf2_constants import Constants
from nvflare.apis.fl_context import FLContext

class Utils():

    def generate_wandb_id(job_name, identity_name):
        return '%s-%s' % (
            job_name,
            identity_name
        )
    
    def generate_job_name(job_id, job_start_timestamp):
        return '%s-%s' % (job_id, job_start_timestamp)
    
    def _get_peer_property(fl_ctx: FLContext, prop, default=None):
        peer_ctx = fl_ctx.get_peer_context()
        value = peer_ctx.get_prop(prop, default)
        if value is None:
            raise Exception('%s is not set' % prop)
        return value
    
    def _set_peer_property(fl_ctx: FLContext, prop, val, private=False, sticky=True):
        peer_ctx = fl_ctx.get_peer_context()
        peer_ctx.set_prop(prop, val, private=private, sticky=sticky)
    
    def get_peer_job_name(fl_ctx: FLContext, default=None):
        return Utils._get_peer_property(fl_ctx, Constants.JOB_NAME, default=default)

    def get_peer_job_start_timestamp(fl_ctx: FLContext, default=None):
        return Utils._get_peer_property(fl_ctx, Constants.JOB_START_TIMESTAMP, default=default)