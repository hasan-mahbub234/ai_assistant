"""initial migration

Revision ID: 96160f7e90df
Revises: 
Create Date: 2025-12-28 10:30:00
"""

# revision identifiers, used by Alembic
revision = "96160f7e90df"
down_revision = None
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def upgrade():
    # Create ENUM types
    user_role = postgresql.ENUM(
        'SUPER_ADMIN', 'ADMIN', 'LAWYER', 'PARALEGAL', 'CLIENT',
        name='user_role'
    )
    user_role.create(op.get_bind())
    
    user_status = postgresql.ENUM(
        'ACTIVE', 'INACTIVE', 'SUSPENDED', 'PENDING',
        name='user_status'
    )
    user_status.create(op.get_bind())
    
    document_status = postgresql.ENUM(
        'PENDING', 'PROCESSING', 'PROCESSED', 'FAILED',
        name='documentstatus'
    )
    document_status.create(op.get_bind())
    
    # ========== USERS TABLE ==========
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('role', user_role, nullable=False, server_default='CLIENT'),
        sa.Column('status', user_status, nullable=False, server_default='PENDING'),
        sa.Column('phone', sa.String(length=20), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('is_email_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('email_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_email_status', 'users', ['email', 'status'])
    op.create_index('ix_users_role_status', 'users', ['role', 'status'])
    
    # ========== REFRESH TOKENS TABLE ==========
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(length=500), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )
    
    op.create_index('ix_refresh_tokens_token', 'refresh_tokens', ['token'])
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])
    
    # ========== API KEYS TABLE ==========
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('scopes', sa.Text(), nullable=False, server_default='read'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash')
    )
    
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])
    
    # ========== PASSWORD RESET TOKENS TABLE ==========
    op.create_table(
        'password_reset_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )
    
    op.create_index('ix_password_reset_tokens_token', 'password_reset_tokens', ['token'])
    op.create_index('ix_password_reset_tokens_user_id', 'password_reset_tokens', ['user_id'])
    
    # ========== DOCUMENTS TABLE ==========
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('document_metadata', sa.JSON(), nullable=True, server_default='{}'),
        sa.Column('status', document_status, nullable=False, server_default='PENDING'),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('ai_analysis', sa.JSON(), nullable=True, server_default='{}'),
        sa.Column('is_encrypted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('encryption_key', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('ix_documents_user_id_status', 'documents', ['user_id', 'status'])
    op.create_index('ix_documents_created_at', 'documents', ['created_at'])
    op.create_index('ix_documents_original_filename', 'documents', ['original_filename'])
    
    # ========== DOCUMENT EMBEDDINGS TABLE ==========
    op.create_table(
        'document_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('pinecone_id', sa.String(length=255), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('chunk_start', sa.Integer(), nullable=False),
        sa.Column('chunk_end', sa.Integer(), nullable=False),
        sa.Column('embedding_model', sa.String(length=100), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('ix_document_embeddings_document_id', 'document_embeddings', ['document_id'])
    op.create_index('ix_document_embeddings_pinecone_id', 'document_embeddings', ['pinecone_id'])
    op.create_index('ix_document_embeddings_chunk_index', 'document_embeddings', ['chunk_index'])
    
    # ========== CHAT HISTORY TABLE ==========
    op.create_table(
        'chat_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('model_used', sa.String(length=100), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('context_chunks', sa.JSON(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('is_helpful', sa.Boolean(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('ix_chat_history_document_id_user_id', 'chat_history', ['document_id', 'user_id'])
    op.create_index('ix_chat_history_created_at', 'chat_history', ['created_at'])
    
    # ========== DOCUMENT SHARES TABLE ==========
    op.create_table(
        'document_shares',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('shared_by_user_id', sa.Integer(), nullable=False),
        sa.Column('shared_with_user_id', sa.Integer(), nullable=False),
        sa.Column('can_view', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('can_edit', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('can_delete', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('can_share', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shared_by_user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shared_with_user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('ix_document_shares_document_id', 'document_shares', ['document_id'])
    op.create_index('ix_document_shares_shared_with_user_id', 'document_shares', ['shared_with_user_id'])
    op.create_index('ix_document_shares_shared_by_user_id', 'document_shares', ['shared_by_user_id'])
    op.create_index('ix_document_shares_expires_at', 'document_shares', ['expires_at'])


def downgrade():
    # Drop tables in reverse order
    op.drop_index('ix_document_shares_expires_at', table_name='document_shares')
    op.drop_index('ix_document_shares_shared_by_user_id', table_name='document_shares')
    op.drop_index('ix_document_shares_shared_with_user_id', table_name='document_shares')
    op.drop_index('ix_document_shares_document_id', table_name='document_shares')
    op.drop_table('document_shares')
    
    op.drop_index('ix_chat_history_created_at', table_name='chat_history')
    op.drop_index('ix_chat_history_document_id_user_id', table_name='chat_history')
    op.drop_table('chat_history')
    
    op.drop_index('ix_document_embeddings_chunk_index', table_name='document_embeddings')
    op.drop_index('ix_document_embeddings_pinecone_id', table_name='document_embeddings')
    op.drop_index('ix_document_embeddings_document_id', table_name='document_embeddings')
    op.drop_table('document_embeddings')
    
    op.drop_index('ix_documents_original_filename', table_name='documents')
    op.drop_index('ix_documents_created_at', table_name='documents')
    op.drop_index('ix_documents_user_id_status', table_name='documents')
    op.drop_table('documents')
    
    op.drop_index('ix_password_reset_tokens_user_id', table_name='password_reset_tokens')
    op.drop_index('ix_password_reset_tokens_token', table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
    
    op.drop_index('ix_api_keys_user_id', table_name='api_keys')
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_table('api_keys')
    
    op.drop_index('ix_refresh_tokens_user_id', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_token', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    
    op.drop_index('ix_users_role_status', table_name='users')
    op.drop_index('ix_users_email_status', table_name='users')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
    
    # Drop ENUM types
    op.execute("DROP TYPE IF EXISTS documentstatus")
    op.execute("DROP TYPE IF EXISTS user_status")
    op.execute("DROP TYPE IF EXISTS user_role")